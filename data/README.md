### Prepare the Data
- Organize dataset-related information into a JSON file with following structure:
   ```json
   {
    "metadata": {
    },
    "data_list": [
        {
            "case_name": <case-name>,
            "RGB_image_path": <RGB-image-path>,
            "LSI_MS_image_path": <LSI-MS-image-path>,
            "MTT_image_path": <MTT-image-path>,
            "vessel_mask_path": <vessel-mask-path>
        },
        ...
    ]
    }

- During training, the class `DataMapper` will parse each data entry in the JSON file and load data.
