## NullClass Task5

This task is about creating a feature to translate the English word to Hindi and it should not translate if the English starts with vowels and other words it should convert . If we enter a English word starts with Vowels it should show an error message as This word starts with Vowels provide some other words and this model should be able to convert english word starts with vowels around 9 PM to 10 PM. The model and its configurations are stored in `english_to_hindi_lstm_model`. The tokenziers are stored in `english_tokenizer.json` and `hindi_tokenizer.json`.

In order to run the notebook, follow the steps:

1. Create a conda environment

```bash
conda create --name nullclass python=3.9
```
2. Activate the environment

```bash
conda activate nullclass
```
3. Install cudnn plugin
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

4. Install tensorflow
```bash
pip install --upgrade pip
# Anything above 2.10 is not supported on the GPU on Windows Native
pip install "tensorflow<2.11" 
```

The same environment `nullclass` can be used for running notebooks for other tasks as well. Now run the notebook named `task5.ipynb`. The GUI for the task 5 is implemented in the file named `gui_task5.py`. 

Run the following command to launch the GUI:
```bash
python gui_task5.py
```

