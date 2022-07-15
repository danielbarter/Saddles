{
  description = "A Python library for computing saddle points built with JAX";

  inputs.nixpkgs.url = github:NixOS/nixpkgs/nixos-21.11;

  outputs = { self, nixpkgs }:
    let commonPackages = pkgs: [
          pkgs.ffmpeg
          pkgs.python3
          pkgs.python3Packages.jax
          pkgs.python3Packages.matplotlib
          pkgs.pyright
        ]; in {

          # nix develop .#gpu
          devShells.x86_64-linux.gpu =
            with import nixpkgs { system = "x86_64-linux"; };
            mkShell {
              buildInputs = with pkgs; [
                (python3Packages.jaxlib.override { cudaSupport = true; })
                nvtop
              ] ++ (commonPackages pkgs);

              LD_LIBRARY_PATH = with pkgs; builtins.concatStringsSep ":" [
                # cuda shared libraries
                "${cudatoolkit_11_2}/lib"
                "${cudatoolkit_11_2.lib}/lib"

                # nvidia driver shared libs
                "${pkgs.linuxPackages.nvidia_x11}/lib"
              ];
            };

          # nix develop .#cpu
          devShells.x86_64-linux.cpu =
            with import nixpkgs { system = "x86_64-linux"; };
            mkShell {
              buildInputs = with pkgs; [
                python3Packages.jaxlib
              ] ++ (commonPackages pkgs);
            };
        };
}
