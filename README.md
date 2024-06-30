# Leaf Disease Detection

This project uses deep learning with a CNN to detect and classify diseases in plant leaves from images. It utilizes the PlantVillage dataset, which includes labeled images of healthy and diseased crop leaves. A Flask-based web application is also provided for easy model interaction.

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/aymanrgab/leaf-disease-detection.git
   cd leaf-disease-detection
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. To preprocess data:
   ```
   python src/data_preprocessing.py
   ```

2. To train the model:
   ```
   python src/train.py
   ```

3. To make predictions:
   ```
   python src/predict.py path/to/image.jpg
   ```

4. To run the web app:
   ```
   python app/main.py
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.