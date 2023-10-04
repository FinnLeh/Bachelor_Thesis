import cv2
import numpy as np
import random
import geopandas as gpd
from osgeo import gdal


def save_as_geotiff(filename, data, geo_transform, projection, region_name=None, mpi_value=None):
    """
    Save a 2D or 3D numpy array as a GeoTIFF file.
    """
    driver = gdal.GetDriverByName('GTiff')
    shape = data.shape
    if len(shape) == 2:
        ds = driver.Create(filename, shape[1], shape[0], 1, gdal.GDT_Float32)
        ds.GetRasterBand(1).WriteArray(data)
    elif len(shape) == 3:
        ds = driver.Create(
            filename, shape[2], shape[1], shape[0], gdal.GDT_Float32)
        for i in range(shape[0]):
            ds.GetRasterBand(i + 1).WriteArray(data[i])
    else:
        raise ValueError("Unsupported array shape for GeoTIFF export")

    ds.SetGeoTransform(geo_transform)
    ds.SetProjection(projection)

    # Hinzufügen von benutzerdefinierten Metadaten (falls bereitgestellt)
    metadata = {}
    if region_name:
        metadata["REGION_NAME"] = region_name
    if mpi_value is not None:
        metadata["MPI_VALUE"] = str(mpi_value)
    if metadata:
        ds.SetMetadata(metadata)

    ds.FlushCache()
    ds = None


def geo_to_pixel(geo_coord, geo_transform):
    """
    Converts geographical coordinates to pixel coordinates.
    """
    x_geo, y_geo = geo_coord
    x_pixel = int((x_geo - geo_transform[0]) / geo_transform[1])
    y_pixel = int((y_geo - geo_transform[3]) / geo_transform[5])
    return x_pixel, y_pixel


def resize_image(image, width):
    aspect_ratio = width / float(image.shape[1])
    height = int(image.shape[0] * aspect_ratio)
    resized = cv2.resize(image, (width, height))
    return resized

# TIFF-Bild laden
ds = gdal.Open('C:/Users/finnl/Downloads/Burundi_Sentinel2_Merged.tif')
image = ds.ReadAsArray()
# Die Bänder von der ersten zur letzten Dimension verschieben
image = np.moveaxis(image, 0, -1)

# get the geotransform
geo_transform = ds.GetGeoTransform()

# Shapefile mit den Regionen laden
regions = gpd.read_file('C:/Users/finnl/Downloads/bdi_admbnda_adm1_igebu_ocha_20171103.shp')

# GeoJSON-Datei mit den MPI-Daten laden
mpi_data = gpd.read_file(
    'C:/Users/finnl/Downloads/Burundi_MPI_Merged_Data.geojson')

# Leere Listen für die ausgewählten Regionen und Labels
selected_regions = []
labels = []
selected_regions_coordinates = []

# Für jede Region
for index, region in regions.iterrows():
    print('Processing region', region['admin1Name'])
    # Erstellen Sie eine Maske für die Region
    mask = np.zeros_like(image[:, :, 0])
    region_coordinates = np.array(region.geometry.exterior.coords.xy).T

    # convert the geographical coordinates to pixel coordinates
    region_coordinates_pixel = np.array(
        [geo_to_pixel(coord, geo_transform) for coord in region_coordinates])

    # create the mask using pixel coordinates
    cv2.fillPoly(mask, [region_coordinates_pixel.astype(int)], 255)

    print(np.unique(mask))
    # display the mask
    # resized_mask = resize_image(mask, 1920)
    # cv2.imshow('Mask', resized_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # MPI-Wert der Region extrahieren
    mpi_value = mpi_data.loc[mpi_data['Sub-national region'] == region['admin1Name'], 'MPI of the region'].values[0]

    selected_regions_coordinates_within_loop = []

    for i in range(24):
        print('Selecting region', i+1)
        while True:
            x = random.randint(0, mask.shape[1] - 40)
            y = random.randint(0, mask.shape[0] - 40)
            if np.all(mask[y:y+40, x:x+40] == 255):
                print('Found suitable region')
                break
            # print('Found a region that is not suitable, trying again')
        region_image = image[y:y+40, x:x+40]
        selected_regions.append(region_image)
        selected_regions_coordinates.append((x, y))
        selected_regions_coordinates_within_loop.append((x, y))
        labels.append(mpi_value)
        

    # # Die Grenzen der Region und die ausgewählten Regionen auf dem Bild anzeigen
    # display_image = image.copy()
    # cv2.polylines(display_image, [region_coordinates_pixel.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=5)
    # for i in range(24):  # Anzahl der ausgewählten Regionen
    #     x, y = selected_regions_coordinates_within_loop[i]
    #     w, h = 40, 40
    #     cv2.rectangle(display_image, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # # Ausschnitt des Bildes anzeigen
    # x, y, w, h = cv2.boundingRect(region_coordinates_pixel.astype(np.int32))
    # zoomed_in_image = display_image[y:y+h, x:x+w]
    # resized_image = resize_image(zoomed_in_image, 1020)
    # cv2.imshow('Selected Regions', resized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# # Die ausgewählten Regionen und die zugehörigen MPI-Werte in einer Datei speichern
# np.savez('C:/Users/finnl/Downloads/selected_regions.npz',
#          images=np.array(selected_regions), labels=np.array(labels))

# # Load the npz file
# loaded = np.load('C:/Users/finnl/Downloads/selected_regions.npz')
# images = loaded['images']
# labels = loaded['labels']

print(selected_regions_coordinates)
print(labels)
print(regions)

# Get the projection from the original TIFF
projection = ds.GetProjection()

# Statt eines festen Wertes für die Schleifenlänge verwenden Sie die tatsächliche Länge der Liste
for j, image in enumerate(selected_regions):
    x, y = selected_regions_coordinates[j]
    adjusted_geotransform = list(geo_transform)
    adjusted_geotransform[0] += x * geo_transform[1]
    adjusted_geotransform[3] += y * geo_transform[5]

    region_index = j // 24
    region_name = regions.iloc[region_index]['admin1Name']
    # da wir die labels-Liste im vorherigen Schritt mit MPI-Werten gefüllt haben
    mpi_value = labels[j]

    # Change the order of the dimensions to (Bands, Height, Width)
    image = np.moveaxis(image, -1, 0)
    save_as_geotiff(
        f'C:/Users/finnl/Downloads/regional_tifs_with_attributes/region_{j}.tif',
        image,
        adjusted_geotransform,
        projection,
        region_name=region_name,
        mpi_value=mpi_value)
