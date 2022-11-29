## Environment

For the frontend environment
```{bash}
cd PANACEA/frontend
npm i
```

For the backend environment:
- edit the file nlisan_env2.yml and modify the last line to point to your conda path
- download pre-trained models nli-san_simpl4vp1.pt and datasets index_data_sources_paragraphs_VP1_B from https://zenodo.org/record/6493847#.Y4aCZ3bP1aY
- execute:
```{bash}
conda env create -n nlisan -f=nlisan_env2.yml
conda activate nlisan
```

## Running
Run frontend:
```{bash}
cd /frontend
npm run dev
```

Run backend:
```{bash}
conda activate nlisan
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```
