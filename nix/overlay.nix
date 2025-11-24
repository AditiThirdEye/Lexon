final: prev:
{
  # Add an alias here can help future migration
  llvmPkgs = final.llvmPackages_17;
  # Use clang instead of gcc to compile, to avoid gcc13 miscompile issue.
  lexon-llvm = final.callPackage ./lexon-llvm.nix { stdenv = final.llvmPkgs.stdenv; };
  lexon-mlir = final.callPackage ./lexon-mlir.nix { stdenv = final.llvmPkgs.stdenv; };
}
