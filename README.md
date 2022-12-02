## Environment

### For the backend environment
Install Required Python Modules
- edit the file nlisan_env2.yml and modify the last line to point to your conda path
- execute:
```{bash}
cd PANACEA
conda env create -n nlisan -f=nlisan_env2.yml
conda activate nlisan
pip install -r requirements.txt
```

### For the frontend environment
Install Node Modules
```{bash}
cd PANACEA/frontend
npm i
```

## Running

### Initialilse and Start Web Server
Run backend:
```{bash}
conda activate nlisan
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```

### Compile the Front-End
Run frontend for development
```{bash}
npm run dev
```
Run the production compile script
```{bash}
npm run build
```