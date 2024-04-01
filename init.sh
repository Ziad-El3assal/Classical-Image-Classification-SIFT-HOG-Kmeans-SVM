Bash
python3 -m venv venv

source venv/bin/activate

./kaggle/kaggle.json
pip install scikit-learn opencv-python pandas numpy kaggle
kaggle datasets download -d puneet6060/intel-image-classification
Expand-Archive intel-image-classification.zip
rm intel-image-classification.zip