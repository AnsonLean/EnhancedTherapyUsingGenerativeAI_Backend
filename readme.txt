It will be best to install everything in a virtual enviroment. 
To do so, you can use the code: "python -m venv virtualEnv"

To activate the virtual enviroment, you can use the following code depending on your operating system"
Window: .\virtualEnv\Scripts\activate
Mac: source ./virtualEnv/bin/activate

Then install the requirements using: 
pip install -r requirements.txt
and 
pip  install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu

After installation is done, you can then use software. Follow the step below to use the software

1. Activate virtual enviroment
Window: .\virtualEnv\Scripts\activate
Mac: source ./virtualEnv/bin/activate

2. Change directory to djangoRest
cd .\djangoRest\ 

3. Set up django
python manage.py migrate

4.Run application
python manage.py runserver