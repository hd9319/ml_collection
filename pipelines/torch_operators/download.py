import os
import json
import csv
import shutil

import requests

def download_pokemon_data(attribute_path, image_directory):
	response = requests.get('http://www.pokemon-data.com')
	if response.status_code == 200:
		pokemons, images = parse_pokemon_data(response.json())

		with open(attribute_path, 'w+') as outfile:
		    writer = csv.writer(outfile)
		    writer.writerows(pokemons)

		for pokemon, image_url in zip(pokemons, images):
			_ = download_image(image_url=image_url,
								image_directory=os.path.join(image_directory,
													str(pokemons[0]) + '.jpg')
								)

	else:
		print('Failed to access webpage.')

def parse_pokemon_data(pokemon_dict):
	data = []
	images = []
	for pokemon in pokemon_dict['pokemon_list']:
		try:
			name = pokemon['name']
			element_type = pokemon['type1']
			image_url = pokemon['image']
			data.append([name, element_type])
			images.append(image_url)

		except ValueError as e:
			print(e)
			continue

	return data, images

def download_image(image_url, image_path):
	r = requests.get(image_url, stream=True)
	if r.status_code == 200:
		with open(image_name, 'wb') as f:
			r.raw.decode_content = True
			shutil.copyfileobj(r.raw, f)

