# Neural Network for the Detection of Destroyed or Damaged Villages in Darfur, Sudan

In this project, coordinating with the Human Rights Data Science Lab at Michigan State Unviersity, I train a neural network capable of detecting if a village in Darfur, Sudan has been damaged or destroyed via free and publically accessible LANDSAT7 30m resolution remote sensing data.

## Motivation

Organizations investigating human rights abuses are often strapped for both cash and researchers. Traditional investigations often rely on a small team of experts poring over hundreds of square miles of remote sensing data looking for signs of building destruction or burning. Developing an algorithm to automatically scan over an area and identify potential zones of destruction will likely save time and energy and allow organizations to collect a more comprehensive body of evidence.

Furthermore, investigations oriented around individual building level analysis often require high resolution imagery, usually less than 5 meters and often 1 meter. Such data is almost always proprietary and tremendously expensive, locking out smaller organizations from important investigative tools. For example, at the Human Rights Lab, we have great difficulty in finding quality and high definition satellite imagery. Google Earth does have some high definiton available, but it is often too out of date to be useful for ongoing human rights abuse, especially in remote areas. LANDSAT and Sentinel data, on the other hand, is free, easily obtainable, and updated regularly with global coverage. But because these data often have bigger resolutions (30m for LANDSAT and 10m for Sentinel) individual buildings are too small to be detected and so a spectroscopic approach is needed. Such an approach is far more accessible for smaller and volunteer organizations.

## Methodology

In a past investigation in the Human Rights Lab, we collected a vast dataset of villages in Darfur, which includes their coordinates and their status (No damage, damaged, or destroyed). Because this dataset contains sensitive information on vulnerable populations, it is not available in the public repository. The spectroscopic data for each village was extracted via Google Earth Engine. Specifically, I extracted the bands B1(blue), B2(green), B3(red), B4(near infrared), and B5 and B7 (both shortwave infrared). I also computed NBR (Normalized Burn Ratio) and NDVI (Normalized Difference Vegetation Index). As for labelling of the data, I combined damged and destroyed into the same 'damaged' category because the distinction is less important that that of whether violence occured or not. dUsing this data, I then trained a neural network binary classifier with tensorflow.

## Results