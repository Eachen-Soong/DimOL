### Project Structure:
Everything is well-decoupled.


├── neuralop
│   ├── data_generator
│   │   ├── ...
│   ├── datasets
│   │   ├── ...
│   ├── kan
│   │   ├── ...
│   ├── layers
│   │   ├── ...
│   ├── models
│   │   ├── classical_solver.py
│   │   ├── FNO_2D.py
│   │   ├── fnogno.py
│   │   ├── fno.py
│   │   ├── LSM_2D.py
│   │   ├── LSM_3D.py
│   │   ├── LSM_Irregular_Geo.py
│   │   ├── model_dispatcher.py
│   │   ├── modules.py
│   │   ├── new_fno.py
│   │   ├── prod_fno.py
│   │   ├── test1.py
│   │   ├── uno.py
│   │   └── utilities.py
│   └── utils.py
├── runs
├── scripts
│   ├── lightning
│   │   ├── callbacks.py
│   │   └── modules.py


├── data/...
├── lightning_modules
│   ├── __init__.py
│   ├── callbacks.py
│   ├── modules.py
├── src/...
├── README.md
├── scripts
│   ├── train.py
├── setup.py
├── todo_list.md
└── .vscode
    └── settings.json