#!/usr/bin/env python3
"""
MLIR Browser - Interactive IR Visualization Tool

This tool captures IR snapshots after each pass and provides an interactive
web interface to browse through the transformations.

Usage:
    python mlir_browser.py input.mlir --passes "canonicalize,cse,..."
    python mlir_browser.py input.mlir --transform transform.mlir
    
Example for matmul:
    python mlir_browser.py asm_src.mlir \\
        --pre-passes "-air-override-memref-memory-space=scope=func memory-space=1" \\
        --transform transform_aie2p.mlir \\
        --passes "air-wrap-func-with-parallel=loop-bounds=2,4,1;air-par-to-launch=depth=0 has-air-segment=true;canonicalize;cse;air-copy-to-dma" \\
        --output matmul_browser.html --no-server

Aggregate mode (browse existing intermediate files):
    python mlir_browser.py --aggregate /path/to/build_peano/air_project/

Transform log mode (parse transform.print output):
    python mlir_browser.py --transform-log output.log -o transform_browser.html
"""

import argparse
import subprocess
import re
import json
import os
import sys
import shlex
import glob
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import webbrowser
import tempfile
import threading

class IRSnapshot:
    """Represents a single IR state after a pass."""
    def __init__(self, pass_name: str, ir_content: str, index: int, before_after: str = "after"):
        self.pass_name = pass_name
        self.ir_content = ir_content
        self.index = index
        self.before_after = before_after
    
    def to_dict(self):
        return {
            "pass_name": self.pass_name,
            "ir_content": self.ir_content,
            "index": self.index,
            "before_after": self.before_after
        }


# Pipeline order for aggregate mode - based on documented compilation pipeline
PIPELINE_FILE_ORDER = [
    # Stage 3: aircc.py intermediate files
    ("placed.air.mlir", "AIR Placement", "After air-collapse-herd, air-place-herds, air-renumber-dma"),
    ("aie.air.mlir", "AIR to AIE", "After air-to-aie pass (converts AIR constructs to AIE dialect)"),
    ("npu.air.mlir", "NPU Instructions", "After air-opt-shim-dma-bds, air-to-std, airrt-to-npu"),
    
    # Stage 4: aiecc.py intermediate files
    ("input_with_addresses.mlir", "Buffer Addresses", "After INPUT_WITH_ADDRESSES_PIPELINE (vector-to-aievec, objectFifo lowering, buffer addresses)"),
    ("input_physical.mlir", "Physical Routing", "After aie-create-pathfinder-flows (routing)"),
    ("input_physical_with_elfs.mlir", "ELF Injection", "After ELF generation and path injection into MLIR"),
    ("npu_insts.mlir", "NPU Lowering", "After NPU lowering pipeline (aie-materialize-bd-chains, aie-dma-to-npu)"),
    
    # Additional files that may be present
    ("*_input_opt_with_addresses.mlir", "LLVM Lowering", "After AIE_LOWER_TO_LLVM pipeline (unified compilation)"),
]


class DebugIRPassInfo:
    """Information about a pass from the debug_ir pass.log."""
    def __init__(self, pass_num: int, pass_name: str, output_file: str, is_checkpoint: bool = False, checkpoint_name: str = None):
        self.pass_num = pass_num
        self.pass_name = pass_name
        self.output_file = output_file
        self.is_checkpoint = is_checkpoint
        self.checkpoint_name = checkpoint_name


class MLIRBrowser:
    """Main class for capturing and serving IR snapshots."""
    
    def __init__(self, air_opt_path: str = None):
        self.air_opt_path = air_opt_path or self._find_air_opt()
        self.snapshots = []
    
    def _find_air_opt(self) -> str:
        """Find air-opt in common locations."""
        candidates = [
            "/home/jiajli/apps/mlir-air/install/bin/air-opt",
            "air-opt",
            os.path.join(os.path.dirname(__file__), "../../install/bin/air-opt"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate) or self._which(candidate):
                return candidate
        raise RuntimeError("Could not find air-opt. Please specify --air-opt-path")
    
    def _which(self, program):
        """Check if a program exists in PATH."""
        import shutil
        return shutil.which(program)
    
    def _format_pass_arg(self, pass_str: str) -> str:
        """
        Format a pass argument, properly quoting values with spaces.
        
        For example:
          "air-par-to-launch=depth=0 has-air-segment=true" 
          becomes
          "-air-par-to-launch='depth=0 has-air-segment=true'"
        
        Uses single quotes for shell compatibility.
        """
        pass_str = pass_str.strip()
        if not pass_str:
            return None
            
        # Check if it already has the - prefix
        if pass_str.startswith('-'):
            prefix = ''
        else:
            prefix = '-'
        
        # Check if it has pass arguments (contains =)
        if '=' in pass_str:
            # Split on first = to get pass name and arguments
            eq_pos = pass_str.index('=')
            pass_name = pass_str[:eq_pos]
            pass_args = pass_str[eq_pos+1:]
            
            # Use single quotes for values with spaces (better for shell)
            if ' ' in pass_args:
                return f"{prefix}{pass_name}='{pass_args}'"
            else:
                return f'{prefix}{pass_name}={pass_args}'
        else:
            return f'{prefix}{pass_str}'
    
    def run_with_ir_printing(self, input_file: str, passes: str = None, 
                             transform_file: str = None, 
                             pre_passes: str = None) -> list:
        """
        Run air-opt with IR printing enabled and capture snapshots.
        
        Args:
            input_file: Path to input MLIR file
            passes: Semicolon or comma-separated list of passes (e.g., "canonicalize;cse")
            transform_file: Path to transform script (alternative to passes)
            pre_passes: Passes to run before transform (e.g., memory space override)
        
        Returns:
            List of IRSnapshot objects
        """
        self.snapshots = []
        
        # Build command - we'll build the full command string then use shell=True
        cmd_parts = [self.air_opt_path, input_file]
        
        # Add pre-passes if specified
        if pre_passes:
            # Split by semicolon if present (for multiple pre-passes)
            separator = ";" if ";" in pre_passes else None
            if separator:
                pre_pass_list = pre_passes.split(separator)
            else:
                pre_pass_list = [pre_passes]
            
            for p in pre_pass_list:
                formatted = self._format_pass_arg(p)
                if formatted:
                    cmd_parts.append(formatted)
        
        # Add transform if specified
        if transform_file:
            cmd_parts.append(f'--air-transform=filename={transform_file}')

        # Add other passes
        if passes:
            # Use semicolon as separator if present, otherwise comma
            separator = ";" if ";" in passes else ","
            for p in passes.split(separator):
                formatted = self._format_pass_arg(p)
                if formatted:
                    cmd_parts.append(formatted)
        
        # Enable IR printing
        cmd_parts.append("--mlir-print-ir-after-all")
        
        # Build the full command string and run with shell=True to handle quoting
        cmd_str = ' '.join(cmd_parts)
        print(f"Running: {cmd_str}")
        
        # Run and capture output with timeout
        try:
            result = subprocess.run(
                cmd_str,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120  # 120 second timeout for large transforms
            )
        except subprocess.TimeoutExpired:
            print("Error: air-opt timed out after 120 seconds")
            print("Try with a simpler input file or fewer passes")
            return self.snapshots
        
        # Combine stdout and stderr (IR dumps go to stderr typically)
        full_output = result.stderr + "\n" + result.stdout
        
        # Parse the output to extract IR snapshots
        self._parse_ir_dumps(full_output, input_file)
        
        return self.snapshots
    
    def _parse_ir_dumps(self, output: str, input_file: str):
        """Parse the --mlir-print-ir-after-all output to extract snapshots."""
        
        # Add initial IR as first snapshot
        try:
            with open(input_file, 'r') as f:
                initial_ir = f.read()
            self.snapshots.append(IRSnapshot("Initial IR", initial_ir, 0, "initial"))
        except Exception as e:
            print(f"Warning: Could not read initial IR: {e}")
        
        # Pattern to match IR dump headers
        # Format: // -----// IR Dump After <PassDescription> (<pass-name>) //----- //
        # Note: Sometimes starts with single / instead of //
        pattern = r'/+\s*-----//\s*IR Dump (After|Before)\s+([^\(]+?)\s*\(([^)]+)\)\s*//----- //'
        
        # Split by the pattern
        parts = re.split(pattern, output)
        
        # parts will be: [prefix, before_after1, pass_desc1, pass_name1, ir1, ...]
        i = 1
        index = 1
        while i + 3 < len(parts):
            before_after = parts[i].lower()
            pass_desc = parts[i + 1].strip()
            pass_name = parts[i + 2].strip()
            ir_content = parts[i + 3]
            
            # Clean up IR content - get just the module
            ir_content = self._extract_module(ir_content)
            
            if ir_content:
                self.snapshots.append(IRSnapshot(
                    f"{pass_desc} ({pass_name})",
                    ir_content,
                    index,
                    before_after
                ))
                index += 1
            
            i += 4
        
        print(f"Parsed {len(self.snapshots)} IR snapshots")
    
    def _extract_module(self, content: str) -> str:
        """Extract the module definition from IR content."""
        if not content:
            return ""
        
        # Find the start of the module or attribute definitions
        # Sometimes IR has #map, #loc definitions before module
        lines = content.split('\n')
        start_idx = 0
        end_idx = len(lines)
        
        # Find first non-empty, non-whitespace line
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and (stripped.startswith('#') or stripped.startswith('module')):
                start_idx = i
                break
        
        # Find the module block
        module_started = False
        brace_count = 0
        
        for i, line in enumerate(lines[start_idx:], start_idx):
            if 'module' in line and '{' in line:
                module_started = True
            
            if module_started:
                brace_count += line.count('{')
                brace_count -= line.count('}')
                
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        result = '\n'.join(lines[start_idx:end_idx])
        
        # Limit size to avoid huge HTML files
        max_size = 500000  # 500KB per snapshot
        if len(result) > max_size:
            result = result[:max_size] + "\n\n... (truncated, IR too large)"
        
        return result.strip()

    def aggregate_from_directory(self, build_dir: str) -> list:
        """
        Aggregate existing intermediate MLIR files from a build directory.
        
        This reads files like placed.air.mlir, aie.air.mlir, etc. from
        a build_peano/air_project/ directory and presents them in pipeline order.
        
        Args:
            build_dir: Path to the build directory (e.g., build_peano/air_project/)
        
        Returns:
            List of IRSnapshot objects
        """
        self.snapshots = []
        build_path = Path(build_dir)
        
        if not build_path.exists():
            raise RuntimeError(f"Build directory not found: {build_dir}")
        
        print(f"Scanning directory: {build_path}")
        
        # Find all MLIR files in the directory
        all_mlir_files = list(build_path.glob("*.mlir"))
        print(f"Found {len(all_mlir_files)} MLIR files")
        
        # Map files to their pipeline order
        file_map = {}
        for mlir_file in all_mlir_files:
            file_map[mlir_file.name] = mlir_file
        
        index = 0
        processed_files = set()
        
        # Process files in pipeline order
        for pattern, stage_name, description in PIPELINE_FILE_ORDER:
            if '*' in pattern:
                # Glob pattern - find matching files
                matching_files = list(build_path.glob(pattern))
                for mlir_file in matching_files:
                    if mlir_file.name not in processed_files:
                        self._add_file_snapshot(mlir_file, stage_name, description, index)
                        processed_files.add(mlir_file.name)
                        index += 1
            elif pattern in file_map:
                mlir_file = file_map[pattern]
                self._add_file_snapshot(mlir_file, stage_name, description, index)
                processed_files.add(pattern)
                index += 1
        
        # Add any remaining MLIR files that weren't in the predefined order
        for mlir_file in sorted(all_mlir_files, key=lambda f: f.name):
            if mlir_file.name not in processed_files:
                # Skip LLVM IR files
                if mlir_file.suffix == '.mlir':
                    stage_name = mlir_file.stem.replace('_', ' ').title()
                    self._add_file_snapshot(mlir_file, stage_name, f"Additional file: {mlir_file.name}", index)
                    processed_files.add(mlir_file.name)
                    index += 1
        
        print(f"Loaded {len(self.snapshots)} IR snapshots from build directory")
        return self.snapshots
    
    def _add_file_snapshot(self, mlir_file: Path, stage_name: str, description: str, index: int):
        """Add a snapshot from a file."""
        try:
            with open(mlir_file, 'r') as f:
                content = f.read()
            
            # Add file info as a comment header
            header = f"// File: {mlir_file.name}\n// Stage: {stage_name}\n// {description}\n\n"
            
            self.snapshots.append(IRSnapshot(
                f"{stage_name} ({mlir_file.name})",
                header + content,
                index,
                "checkpoint"
            ))
            print(f"  [{index}] {stage_name}: {mlir_file.name}")
        except Exception as e:
            print(f"  Warning: Could not read {mlir_file}: {e}")
    
    def parse_debug_ir_log(self, log_file: Path) -> list:
        """
        Parse the pass.log file from a debug_ir directory.
        
        The log file has entries like:
        [PASS 001] air-insert-launch-around-herd{insert-segment=true}
          -> Output: air_project/debug_ir/pass_001_after_air-insert-launch-around-herd.mlir
        
        And checkpoint markers like:
        ======================================================================
        CHECKPOINT: AIR Placement Complete
          This is equivalent to: placed.air.mlir
          Debug IR file: air_project/debug_ir/pass_042_after_func.func_air-renumber-dma.mlir
        ======================================================================
        
        Args:
            log_file: Path to the pass.log file
        
        Returns:
            List of DebugIRPassInfo objects
        """
        pass_infos = []
        
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Parse pass entries
        # Pattern: [PASS NNN] pass-name{args} or [PASS NNN] [Initial IR before passes]
        #          -> Output: path/to/file.mlir
        pass_pattern = r'\[PASS (\d+)\]\s+(.+?)\n\s*->\s*Output:\s*(.+\.mlir)'
        
        for match in re.finditer(pass_pattern, content):
            pass_num = int(match.group(1))
            pass_name = match.group(2).strip()
            output_file = match.group(3).strip()
            
            pass_infos.append(DebugIRPassInfo(
                pass_num=pass_num,
                pass_name=pass_name,
                output_file=output_file
            ))
        
        # Parse checkpoint markers
        # Each checkpoint block looks like:
        # ======================================================================
        # CHECKPOINT: AIR Placement Complete
        #   This is equivalent to: placed.air.mlir
        #   Debug IR file: air_project/debug_ir/pass_042_after_func.func_air-renumber-dma.mlir
        # ======================================================================
        # Use a more specific pattern to avoid greedy matching
        checkpoint_pattern = r'CHECKPOINT:\s*(.+?)\n[^\n]*\n\s*Debug IR file:\s*(\S+\.mlir)'
        
        for match in re.finditer(checkpoint_pattern, content):
            checkpoint_name = match.group(1).strip()
            debug_file = match.group(2).strip()
            
            # Find the pass info that matches this checkpoint file
            for info in pass_infos:
                if info.output_file.endswith(os.path.basename(debug_file)):
                    info.is_checkpoint = True
                    info.checkpoint_name = checkpoint_name
                    break
        
        return pass_infos
    
    def aggregate_from_debug_ir(self, debug_ir_dir: str) -> list:
        """
        Aggregate IR snapshots from a debug_ir directory generated by DEBUG_AIRCC=1.
        
        This reads the pass.log file to get the correct pass names and order,
        then loads all pass_*.mlir files.
        
        Args:
            debug_ir_dir: Path to the debug_ir directory (e.g., build_peano/air_project/debug_ir/)
        
        Returns:
            List of IRSnapshot objects
        """
        self.snapshots = []
        debug_path = Path(debug_ir_dir)
        
        if not debug_path.exists():
            raise RuntimeError(f"Debug IR directory not found: {debug_ir_dir}")
        
        log_file = debug_path / "pass.log"
        
        # Check if pass.log exists
        if log_file.exists():
            print(f"Parsing pass.log from: {debug_path}")
            pass_infos = self.parse_debug_ir_log(log_file)
            print(f"Found {len(pass_infos)} passes in log")
        else:
            print(f"Warning: pass.log not found in {debug_path}, using file ordering")
            pass_infos = None
        
        if pass_infos:
            # Use pass.log ordering
            for info in sorted(pass_infos, key=lambda x: x.pass_num):
                mlir_file = debug_path / os.path.basename(info.output_file)
                
                if not mlir_file.exists():
                    print(f"  Warning: {mlir_file.name} not found, skipping")
                    continue
                
                try:
                    with open(mlir_file, 'r') as f:
                        content = f.read()
                    
                    # Determine the badge type
                    if info.pass_num == 0:
                        badge = "initial"
                    elif info.is_checkpoint:
                        badge = "checkpoint"
                    else:
                        badge = "after"
                    
                    # Format pass name with checkpoint info if applicable
                    display_name = f"[{info.pass_num:03d}] {info.pass_name}"
                    if info.is_checkpoint:
                        display_name += f" ★ {info.checkpoint_name}"
                    
                    self.snapshots.append(IRSnapshot(
                        display_name,
                        content,
                        info.pass_num,
                        badge
                    ))
                    
                    checkpoint_marker = " ★" if info.is_checkpoint else ""
                    print(f"  [{info.pass_num:03d}] {info.pass_name}{checkpoint_marker}")
                    
                except Exception as e:
                    print(f"  Warning: Could not read {mlir_file}: {e}")
        else:
            # Fallback: sort by filename
            mlir_files = sorted(debug_path.glob("pass_*.mlir"))
            
            for index, mlir_file in enumerate(mlir_files):
                # Extract pass info from filename
                # Format: pass_NNN_after_PASSNAME.mlir or pass_000_initial_input.mlir
                match = re.match(r'pass_(\d+)_(.+)\.mlir', mlir_file.name)
                if match:
                    pass_num = int(match.group(1))
                    pass_name = match.group(2).replace('_', ' ').replace('after ', '')
                else:
                    pass_num = index
                    pass_name = mlir_file.stem
                
                try:
                    with open(mlir_file, 'r') as f:
                        content = f.read()
                    
                    badge = "initial" if pass_num == 0 else "after"
                    
                    self.snapshots.append(IRSnapshot(
                        f"[{pass_num:03d}] {pass_name}",
                        content,
                        pass_num,
                        badge
                    ))
                    print(f"  [{pass_num:03d}] {pass_name}")
                    
                except Exception as e:
                    print(f"  Warning: Could not read {mlir_file}: {e}")
        
        print(f"\nLoaded {len(self.snapshots)} IR snapshots from debug_ir directory")
        return self.snapshots

    def parse_transform_log(self, log_file: str) -> list:
        """
        Parse a log file containing transform.print output.
        
        The transform.print operation outputs markers like:
        [[[ IR printer: <name> ]]]
        <IR content>
        
        This method extracts the IR snapshots from such a log file.
        
        Args:
            log_file: Path to the log file containing transform.print output
        
        Returns:
            List of IRSnapshot objects
        """
        self.snapshots = []
        
        log_path = Path(log_file)
        if not log_path.exists():
            raise RuntimeError(f"Log file not found: {log_file}")
        
        print(f"Parsing transform log: {log_path}")
        
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Pattern to match transform.print markers
        # Format: [[[ IR printer: <name> ]]]
        # Note: The name may contain brackets like [00], so we need to match until ]]]
        pattern = r'\[\[\[ IR printer: (.+?) \]\]\]'
        
        # Split content by the pattern
        parts = re.split(pattern, content)
        
        # parts will be: [prefix, name1, ir1, name2, ir2, ...]
        # The prefix contains any content before the first marker
        
        index = 0
        
        # Check if there's meaningful content before the first marker (initial state)
        prefix = parts[0].strip()
        if prefix and 'module' in prefix.lower():
            # Extract any initial IR that appears before the first transform.print
            initial_ir = self._extract_module_from_text(prefix)
            if initial_ir:
                self.snapshots.append(IRSnapshot(
                    "Initial IR (before transforms)",
                    initial_ir,
                    index,
                    "initial"
                ))
                index += 1
        
        # Process transform.print snapshots
        i = 1
        while i + 1 < len(parts):
            phase_name = parts[i].strip()
            ir_content = parts[i + 1]
            
            # Extract the module from the IR content
            ir_content = self._extract_module_from_text(ir_content)
            
            if ir_content:
                self.snapshots.append(IRSnapshot(
                    phase_name,
                    ir_content,
                    index,
                    "checkpoint"
                ))
                index += 1
                print(f"  [{index}] {phase_name}")
            
            i += 2
        
        print(f"Parsed {len(self.snapshots)} IR snapshots from transform log")
        return self.snapshots
    
    def _extract_module_from_text(self, text: str) -> str:
        """
        Extract MLIR module content from text that may contain other content.
        
        This handles cases where the log contains additional output like
        "Loading input MLIR from:", "PASS!", etc.
        """
        if not text:
            return ""
        
        lines = text.split('\n')
        result_lines = []
        in_module = False
        brace_count = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Skip common log messages
            if any(msg in stripped for msg in [
                'Loading input MLIR',
                'XRTBackend:',
                'PASS!',
                'Running ',
                'Using transform script:',
                'Using input MLIR:',
            ]):
                continue
            
            # Start of module or affine map definitions
            if stripped.startswith('#map') or stripped.startswith('#loc') or stripped.startswith('module'):
                in_module = True
            
            if in_module:
                result_lines.append(line)
                brace_count += line.count('{')
                brace_count -= line.count('}')
                
                # End of module
                if 'module' in line and brace_count == 0 and len(result_lines) > 1:
                    break
        
        result = '\n'.join(result_lines).strip()
        
        # Limit size to avoid huge HTML files
        max_size = 500000  # 500KB per snapshot
        if len(result) > max_size:
            result = result[:max_size] + "\n\n... (truncated, IR too large)"
        
        return result
    
    def export_snapshots(self, output_file: str = "ir_snapshots.json"):
        """Export snapshots to JSON file."""
        data = {
            "snapshots": [s.to_dict() for s in self.snapshots],
            "total_count": len(self.snapshots)
        }
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        return output_file
    
    def generate_html(self, output_file: str = "mlir_browser.html"):
        """Generate a self-contained HTML file with embedded snapshots."""
        script_dir = Path(__file__).parent
        
        # Read the template HTML
        with open(script_dir / "index.html", 'r') as f:
            html_content = f.read()
        
        # Prepare data
        data = {
            "snapshots": [s.to_dict() for s in self.snapshots],
            "total_count": len(self.snapshots)
        }
        json_data = json.dumps(data)
        
        # Check if there's already an embedded snapshots script and replace it
        # Pattern to find existing window.embeddedSnapshots
        existing_pattern = r'<script>window\.embeddedSnapshots = \{[^<]*\};</script>\s*'
        html_content = re.sub(existing_pattern, '', html_content)
        
        # Inject data into HTML
        # We look for the first script tag and inject our data before it
        injection = f'<script>window.embeddedSnapshots = {json_data};</script>\n    '
        
        # Insert before the first script tag
        if '<script>' in html_content:
            html_content = html_content.replace('<script>', injection + '<script>', 1)
        else:
            html_content = html_content.replace('</body>', injection + '</body>')
            
        with open(output_file, 'w') as f:
            f.write(html_content)
            
        return output_file
    
    def serve_browser(self, port: int = 8080):
        """Start the web server and open browser."""
        # Export snapshots
        script_dir = Path(__file__).parent
        self.export_snapshots(script_dir / "ir_snapshots.json")
        
        # Change to script directory to serve files
        os.chdir(script_dir)
        
        handler = SimpleHTTPRequestHandler
        httpd = HTTPServer(('localhost', port), handler)
        
        print(f"\n✨ MLIR Browser is running at http://localhost:{port}/index.html")
        print(f"   Loaded {len(self.snapshots)} IR snapshots")
        print("   Press Ctrl+C to stop\n")
        
        # Open browser in a separate thread
        def open_browser():
            webbrowser.open(f'http://localhost:{port}/index.html')
        
        threading.Timer(0.5, open_browser).start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down MLIR Browser...")
            httpd.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="MLIR Browser - Interactive IR Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with a pass pipeline
  python mlir_browser.py input.mlir --passes "canonicalize;cse"
  
  # Run with a transform script
  python mlir_browser.py input.mlir --transform script.mlir
  
  # Full matmul example
  python mlir_browser.py asm_src.mlir \\
    --pre-passes "-air-override-memref-memory-space=scope=func memory-space=1" \\
    --transform transform_aie2p.mlir \\
    --passes "air-wrap-func-with-parallel=loop-bounds=2,4,1;air-par-to-launch=depth=0 has-air-segment=true;canonicalize;cse;air-copy-to-dma" \\
    --output matmul_browser.html --no-server

  # Aggregate mode - browse existing intermediate files
  python mlir_browser.py --aggregate /path/to/build_peano/air_project/ \\
    --output pipeline_browser.html
"""
    )
    
    parser.add_argument("input", nargs='?', help="Input MLIR file (not required for aggregate mode)")
    parser.add_argument("--passes", "-p", help="Semicolon or comma-separated list of passes")
    parser.add_argument("--transform", "-t", help="Transform script file")
    parser.add_argument("--pre-passes", help="Passes to run before main passes")
    parser.add_argument("--air-opt-path", help="Path to air-opt binary")
    parser.add_argument("--port", type=int, default=8080, help="Port for web server")
    parser.add_argument("--export-only", action="store_true", 
                        help="Only export JSON, don't start server")
    parser.add_argument("--output", "-o", help="Output HTML file path (default: mlir_browser.html)")
    parser.add_argument("--no-server", action="store_true", help="Don't start the web server, just generate HTML")
    
    # Aggregate mode options
    parser.add_argument("--aggregate", "-a", metavar="BUILD_DIR",
                        help="Aggregate mode: read existing intermediate MLIR files from build directory (e.g., build_peano/air_project/)")
    parser.add_argument("--transform-log", "-l", metavar="LOG_FILE",
                        help="Transform log mode: parse transform.print output from a log file")
    parser.add_argument("--debug-ir", "-d", metavar="DEBUG_IR_DIR",
                        help="Debug IR mode: read pass-by-pass IR files from debug_ir directory generated by DEBUG_AIRCC=1")
    
    args = parser.parse_args()
    
    # Debug IR mode - read pass-by-pass IR files from debug_ir directory
    if args.debug_ir:
        try:
            browser = MLIRBrowser()
            browser.aggregate_from_debug_ir(args.debug_ir)
            
            if len(browser.snapshots) == 0:
                print("Warning: No IR snapshots found in the debug_ir directory.")
                print("Make sure the directory contains pass_*.mlir files.")
                sys.exit(1)
            
            # Generate output
            if args.export_only:
                output = browser.export_snapshots()
                print(f"Exported {len(browser.snapshots)} snapshots to {output}")
            else:
                output_html = args.output or "debug_ir_browser.html"
                browser.generate_html(output_html)
                print(f"\n✨ Generated self-contained HTML: {output_html}")
                print(f"   Open in browser to explore {len(browser.snapshots)} compilation passes")
                
                if not args.no_server:
                    browser.serve_browser(args.port)
                    
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        return
    
    # Transform log mode - parse transform.print output from log file
    if args.transform_log:
        try:
            browser = MLIRBrowser()
            browser.parse_transform_log(args.transform_log)
            
            if len(browser.snapshots) == 0:
                print("Warning: No IR snapshots found in the log file.")
                print("Make sure the log contains transform.print output with markers like:")
                print("  [[[ IR printer: <phase name> ]]]")
                sys.exit(1)
            
            # Generate output
            if args.export_only:
                output = browser.export_snapshots()
                print(f"Exported {len(browser.snapshots)} snapshots to {output}")
            else:
                output_html = args.output or "transform_browser.html"
                browser.generate_html(output_html)
                print(f"\n✨ Generated self-contained HTML: {output_html}")
                print(f"   Open in browser to explore {len(browser.snapshots)} transform phases")
                
                if not args.no_server:
                    browser.serve_browser(args.port)
                    
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        return
    
    # Aggregate mode - read existing files from build directory
    if args.aggregate:
        try:
            browser = MLIRBrowser()
            browser.aggregate_from_directory(args.aggregate)
            
            if len(browser.snapshots) == 0:
                print("Warning: No MLIR files found in the specified directory.")
                sys.exit(1)
            
            # Generate output
            if args.export_only:
                output = browser.export_snapshots()
                print(f"Exported {len(browser.snapshots)} snapshots to {output}")
            else:
                output_html = args.output or "pipeline_browser.html"
                browser.generate_html(output_html)
                print(f"\n✨ Generated self-contained HTML: {output_html}")
                print(f"   Open in browser to explore {len(browser.snapshots)} compilation stages")
                
                if not args.no_server:
                    browser.serve_browser(args.port)
                    
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        return
    
    # Normal mode - run passes and capture IR
    if not args.input:
        print("Error: Please specify an input MLIR file or use --aggregate mode")
        parser.print_help()
        sys.exit(1)
    
    if not args.passes and not args.transform and not args.pre_passes:
        print("Error: Please specify at least one of --passes, --transform, or --pre-passes")
        parser.print_help()
        sys.exit(1)
    
    try:
        browser = MLIRBrowser(args.air_opt_path)
        browser.run_with_ir_printing(
            args.input,
            passes=args.passes,
            transform_file=args.transform,
            pre_passes=args.pre_passes
        )
        
        if len(browser.snapshots) == 0:
            print("Warning: No IR snapshots were captured.")
            print("This might happen if no passes made changes to the IR.")
        else:
            print(f"Captured {len(browser.snapshots)} IR snapshots")
        
        if args.export_only:
            output = browser.export_snapshots()
            print(f"Exported {len(browser.snapshots)} snapshots to {output}")
        else:
            # Generate self-contained HTML
            output_html = args.output or "mlir_browser.html"
            browser.generate_html(output_html)
            print(f"Generated self-contained HTML: {output_html}")
            
            if not args.no_server:
                browser.serve_browser(args.port)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
