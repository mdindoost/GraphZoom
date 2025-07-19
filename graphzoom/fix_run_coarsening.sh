#!/bin/bash
# Fixed version of run_coarsening.sh for MCR 2018a with v94

# Backup original
cp run_coarsening.sh run_coarsening.sh.backup

# Create fixed version
cat > run_coarsening.sh << 'SCRIPT_EOF'
#!/bin/sh
# script for execution of deployed applications
#
# Sets up the MATLAB Runtime environment for the current $ARCH and executes 
# the specified command.
#
exe_name=$0
exe_dir=`dirname "$0"`
echo "------------------------------------------"
if [ "x$1" = "x" ]; then
  echo Usage:
  echo    $0 \<deployedMCRroot\> args
else
  echo Setting up environment variables
  MCRROOT="$1"
  echo ---
  # Updated paths for MCR 2018a with v94 directory structure
  LD_LIBRARY_PATH=.:${MCRROOT}/v94/runtime/glnxa64 ;
  LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MCRROOT}/v94/bin/glnxa64 ;
  LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MCRROOT}/v94/sys/os/glnxa64;
  LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MCRROOT}/v94/sys/opengl/lib/glnxa64;
  LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MCRROOT}/v94/extern/bin/glnxa64;
  export LD_LIBRARY_PATH;
  echo LD_LIBRARY_PATH is ${LD_LIBRARY_PATH};
  shift 1
  args=
  while [ $# -gt 0 ]; do
      token=$1
      args="${args} \"${token}\"" 
      shift
  done
  eval "\"${exe_dir}/coarsening\"" $args
fi
exit
SCRIPT_EOF

# Make executable
chmod +x run_coarsening.sh

echo "✅ Fixed run_coarsening.sh with v94 paths"
echo "📋 Changes made:"
echo "  - Added /v94/ to all MCR library paths"
echo "  - Added extern/bin/glnxa64 path"
echo ""
echo "🔧 Now testing LAMG with fixed script..."
