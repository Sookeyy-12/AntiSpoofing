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

## Collecting Data for Training

1. Navigate to `dataCollecting.py` in the Root Directory.

2. Modify the `classID`.
    - `1` for Real Faces.
    - `0` for Spoof Faces.

3. Run the following command:
    - Before running this command, ensure that your camera is properly set up and that only 1 face is visible in the camera frame at a time along with:
        - Real Faces are visible in the camera frame for `classID=1`.
        - Spoof Faces are visible in the camera frame for `classID=0`.

            ```bash
            python dataCollecting.py
            ```

4. After you run the command, the script will start storing the Images and the corresponding Labels in the `Datasets/DataCollect` directory. Press `q` or `Ctrl+C` to stop the script.

5. Navigate to the `Datasets/DataCollect` directory and move the Images and their Labels to the respective directories in the `Real` or `Fake` directory, depending on your `ClassID`. Then clean up the `DataCollect` directory.

6. Repeat Steps `1-5` for the other `ClassID`.

7. Copy Equal Number of Images-Labels from both the `Real` and `Fake` directories to the `Datasets/All` directory.
    - *Note: Make sure you select the correct combinations of Image and Label. (an Image and its corresponding Label should have the same name)*

8. Navigate to the Root Directory and run the following command:
    ```bash
    python splitData.py
    ```

    - *Note: This will automate splitting of data and create a zip file called `SplitData.zip` inside of  `Datasets` directory.*

9. You can now use this `SplitData.zip` to train the model.

## Training YOLO on the Collected Data

1. In the Root directory, open the `training.ipynb` file in Google Colab.

2. Upload the `SplitData.zip` file to your Google Drive.
    - *Note: You can also upload it directly to the Colab Environment but that would be very slow.*

3. Run the cells in the `training.ipynb` file to train the model.
    - *Note: You can also modify/add hyperparamteres.*

4. After the training is complete, Navigate to `/runs/detect/` to choose the `best` model and download it.