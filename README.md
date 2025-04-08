# Forest Human Nexus

This repository contains all the codes and the link to the datasets used in this research.

All analyses were conducted using the Mollweide 54009 projection with a 1km resolution from 1975 to 2020 every 5 years. EPSG:54009 refers to the Mollweide projection in the EPSG (European Petroleum Survey Group) database, a widely used spatial reference system. The Mollweide projection is an equal-area map projection, commonly used for world maps where preserving area relationships is important (e.g., for thematic mapping of global data such as climate, population, or vegetation). 

### 1. Forest data manipulation
1. The first step is to create a ternary forest dataset from the HILDA+ original data. 0 Backgroud, 1 No Forest, 2 Forest
    - 1.1 We convert the HILDA+ dataset to the mollweide projection in the code named **toMollweide.py**
    - 1.2 The code to create the ternary forest data is in the folder *codes* and it is named **ternaryForest.py**
    - 1.3 Run the GUIDOS workbench to create the distance to forest dataset. The code for this is **forestGuidos.py** (read Guidos toolbox instructions https://forest.jrc.ec.europa.eu/en/activities/lpa/gwb/)

### 2. Forest Area per Capita
The Forest Area per Capita (FAC) dataset is created using both the HILDA+ and the population datasets from EU Global Human Settlement Layers (https://human-settlement.emergency.copernicus.eu/). The code to compute the FAC indicator is named **FAC.py**.
