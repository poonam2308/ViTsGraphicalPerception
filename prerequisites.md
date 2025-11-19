### Option 1: Conda installation 

#### Download the Miniconda installer
Download the [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer)
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

```
#### Run the installer 
```bash
bash Miniconda3-latest-Linux-x86_64.sh
```
Then follow the prompts:
`Accept the license (yes)
Choose install location 
When asked “Do you wish the installer to initialize Miniconda3 by running conda init?” → say yes.`


#### Refresh the shell (depending on your shell)
```bash
source ~/.bashrc
```

#### Verify the conda version
```bash 
conda list
```

After verfiying the version, first Installation ends here. 



> **Note (if `conda` is not found)**  
> If your shell says `conda: command not found` after installing Miniconda/Anaconda, run the following once:
>
> ```bash
> # Adjust the path if you installed Miniconda/Anaconda somewhere else
> source ~/miniconda3/etc/profile.d/conda.sh
> conda init bash
> exec $SHELL
> ```
>
> After this, `conda activate vitsgp` should work from any directory.
