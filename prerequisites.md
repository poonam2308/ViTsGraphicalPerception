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
