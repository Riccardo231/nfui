"""
Helper module for compiling and running Fortran neural networks with neural-fortran
"""

import os
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

class FortranRunner:
    """Class to handle compilation and execution of neural-fortran code"""
    
    def __init__(self, callback=None):
        """Initialize runner with optional callback for output streaming"""
        self.callback = callback  # Function to call with output lines
        self.process = None
        self.temp_dir = None
    
    def check_neural_fortran(self):
        """Check if neural-fortran is available on the system"""
        try:
            # Try to compile a minimal program that links to neural-fortran
            with tempfile.NamedTemporaryFile(suffix='.f90', delete=False) as f:
                f.write(b"""program check_nf
  use nf, only: network
  implicit none
  print *, 'neural-fortran is available'
end program check_nf
""")
                test_file = f.name
                
            # Try to compile
            compile_cmd = f"gfortran -o check_nf {test_file} -lneural-fortran"
            result = subprocess.run(compile_cmd, shell=True, 
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Clean up the test file
            try:
                os.unlink(test_file)
                if os.path.exists("check_nf"):
                    os.unlink("check_nf")
            except:
                pass
                
            return result.returncode == 0
            
        except Exception as e:
            print(f"Error checking neural-fortran: {e}")
            return False
    
    def compile_and_run(self, fortran_code, program_name="nn_model"):
        """Compile and run a Fortran neural network program"""
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        try:
            # Write the Fortran code to a file
            program_file = os.path.join(self.temp_dir, f"{program_name}.f90")
            with open(program_file, 'w') as f:
                f.write(fortran_code)
            
            # Compile the code
            executable = os.path.join(self.temp_dir, program_name)
            compile_cmd = f"gfortran -o {executable} {program_file} -lneural-fortran"
            
            self._output_callback(f"Compiling Fortran code...\n")
            self._output_callback(f"$ {compile_cmd}\n")
            
            compile_process = subprocess.Popen(
                compile_cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # Stream compile output
            self._stream_output(compile_process)
            
            # Check compilation result
            if compile_process.returncode != 0:
                self._output_callback("Compilation failed!\n")
                return False
            
            # Run the executable
            self._output_callback(f"Running neural network...\n")
            self._output_callback(f"$ {executable}\n")
            
            run_process = subprocess.Popen(
                executable, shell=True, cwd=self.temp_dir,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # Stream run output
            self._stream_output(run_process)
            
            # Return success or failure
            if run_process.returncode == 0:
                self._output_callback("Neural network execution completed successfully.\n")
                return True
            else:
                self._output_callback("Neural network execution failed.\n")
                return False
                
        except Exception as e:
            self._output_callback(f"Error: {e}\n")
            return False
        finally:
            # Clean up temporary directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
    
    def _stream_output(self, process):
        """Stream output from a subprocess"""
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                self._output_callback(output)
        
        return process.poll()
    
    def _output_callback(self, message):
        """Send output to callback function if available"""
        if self.callback:
            self.callback(message)
        else:
            print(message, end='')
            sys.stdout.flush()

if __name__ == "__main__":
    # Simple test when run directly
    runner = FortranRunner()
    if runner.check_neural_fortran():
        print("neural-fortran is available on this system.")
    else:
        print("neural-fortran is NOT available. Please install it first.")
