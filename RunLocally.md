# <p align="center">Running the Project Locally</p>

<p align="center"> STEP 0: Star this Repository ⭐! (optional) </p>


## Table of Contents

- [Setting up Environment](#setting-up-the-environment)
- [Inference](#inference)

## Setting up the Environment

1. **Clone the repository**

    ```bash
    git clone <repo url>
    ```
2. **Set up the Environment**
    
    - Navigate to the cloned repository.
        ```bash
        cd AntiSpoofing
        ```
    - Create a virtual environment.
        ```bash
        python -m venv <env_name>
        ```
    - Activate the virtual environment.
        ```bash
        <env_name>\Scripts\activate
        ```
    - *Note: To deactivate the environment, run `deactivate` in the terminal.*

3. **Install the Required Packages**
    
    ```bash
    pip install -r requirements.txt
    ```
     - *Note: Make sure to run this command inside the virtual environment.*



## Inference

1. In the Root Directory of the Repository, run the following command:

    ```bash
    python main.py
    ```

    - *Note: In the early stages of this Project, you might need to change the `confidence` value in the `main.py` or **Train** a new model altogether for this to work.*

2. To **Exit** the Inference window, either press `q` or `Ctrl+C` in the terminal.