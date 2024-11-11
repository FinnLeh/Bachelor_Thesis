import os
import numpy as np
import rasterio
from PIL import Image


def tiffs_to_grayscale_npy(path_to_files, save_path):
    """
    Lädt TIFF-Dateien von einem gegebenen Pfad, konvertiert sie in Graustufen und speichert sie in einem .npy-File.
    
    Parameters:
    - path_to_files: Pfad zum Verzeichnis, das die TIFF-Dateien enthält.
    - save_path: Pfad, an dem das .npy-File gespeichert werden soll.
    """

    image_files = [os.path.join(path_to_files, file) for file in os.listdir(
        path_to_files) if file.endswith('.tif')]

    grayscale_images = []

    for file in image_files:
        # Bild mit rasterio öffnen
        with rasterio.open(file) as dataset:
            # PIL-Bild aus Array erzeugen
            img_array = dataset.read().transpose((1, 2, 0))
            pil_img = Image.fromarray((img_array * 255).astype(np.uint8))

            # PIL-Bild in Graustufen konvertieren
            grayscale_img = pil_img.convert("L")
            grayscale_images.append(np.array(grayscale_img))

    # Liste von Graustufenbildern in ein numpy Array umwandeln
    grayscale_array = np.array(grayscale_images)

    # Daten in .npy-File speichern
    np.save(save_path, grayscale_array)


# Pfad zu Ihren TIFF-Dateien
path_to_files = 'c:/Users/finnl/Downloads/regional_tifs_with_attributes'
save_path = 'c:/Users/finnl/Downloads/X_Burundi.npy'

tiffs_to_grayscale_npy(path_to_files, save_path)
