{
  description = "A Python library for computing saddle points built with JAX";

  inputs.nixpkgs.url = github:NixOS/nixpkgs/nixos-21.11;

  outputs = { self, nixpkgs }: {

    # nix develop .#gpu
    devShells.x86_64-linux.gpu =
      with import nixpkgs { system = "x86_64-linux"; };
      mkShell {
        buildInputs = with pkgs; [
          ffmpeg
          python3
          python3Packages.jax
          (python3Packages.jaxlib.override { cudaSupport = true; })
          python3Packages.matplotlib
          nvtop
        ];

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
          ffmpeg
          python3
          python3Packages.jax
          python3Packages.jaxlib
          python3Packages.matplotlib
        ];
      };
  };
}
