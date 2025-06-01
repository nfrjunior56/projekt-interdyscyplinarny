import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
# ---------------------------------------------
import pandas as pd 
# Import biblioteki do pracy z danymi w formie tabelii
# ---------------------------------------------
from shapely.geometry import box, Polygon
from shapely.wkt import loads
from skimage.filters import threshold_otsu
from eolearn.core import EOPatch, FeatureType
from sentinelhub import CRS, BBox, DataCollection, SHConfig
from eolearn.core import EOTask, EOWorkflow, OutputTask, linearly_connect_tasks
from eolearn.features import NormalizedDifferenceIndexTask, SimpleFilterTask
from eolearn.geometry import VectorToRasterTask
from eolearn.io import SentinelHubInputTask
from skimage.measure import find_contours
# ---------------------------------------------
# ETAP 4 - MACHINE LEARNING
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
# ---------------------------------------------


# ---------------------------------------------
# WKT - Jezioro Łukajno (obszar zainteresowania)
# ---------------------------------------------
wkt_src = "POLYGON ((21.6363969956964 53.80115266656827, 21.637368658330928 53.80194722568004, 21.637659326640033 53.802226789194236, 21.63876779090583 53.804450311707825, 21.639149812112265 53.805014307775735, 21.639996902613774 53.80578917695746, 21.641242623845898 53.80681414341561, 21.641815655655506 53.80709367447997, 21.642189372053338 53.807383011688955, 21.64202327587725 53.80814802926557, 21.642031580685796 53.808388320366646, 21.642247505716142 53.80867764864038, 21.64262040373356 53.80889925997829, 21.64305135444232 53.80990596759156, 21.642968953539764 53.81097257276443, 21.64300210082709 53.81111609943929, 21.643200984550873 53.811403151315716, 21.643692669739977 53.812114928472624, 21.64388050436827 53.81235304551657, 21.644327547314816 53.81370850786317, 21.644377268245393 53.81392704538533, 21.644647971092212 53.814338023310995, 21.645322545943344 53.814704700178794, 21.645904771448187 53.81511897552974, 21.64728284401619 53.816367605635634, 21.64774994291119 53.81685219217937, 21.64782176203434 53.81716855822262, 21.647606304666567 53.81765777936042, 21.647767166893857 53.8179444030541, 21.647745068702108 53.81813682715432, 21.647176040269642 53.81826076085218, 21.646548057569277 53.8189874548512, 21.646459664802876 53.81936576994423, 21.646227633791057 53.819577755357074, 21.645354404703227 53.82115583038396, 21.644851670845384 53.8217656671618, 21.644431805205215 53.8221928737359, 21.64317953815197 53.822568697187535, 21.642047005834 53.82290458636092, 21.64083712892301 53.82335607728524, 21.639516761976807 53.82385827008167, 21.639461516498585 53.82388109688378, 21.637991986759147 53.8241974098633, 21.637534649310055 53.824294525699344, 21.63652365704658 53.82425539441692, 21.63542427201594 53.82464018377405, 21.634742536672377 53.82476048275046, 21.63442763744277 53.82514200659628, 21.633456520946368 53.82526520676947, 21.63185765478758 53.82510103503441, 21.630100484671914 53.82519502101803, 21.629001099642096 53.825498280700174, 21.626824426060892 53.82569718991496, 21.625288984532972 53.82540935876014, 21.62412330492768 53.82540935876014, 21.623449310084823 53.82570935606239, 21.622266790425158 53.825647054348394, 21.622570640559246 53.82541227368472, 21.62225574132961 53.82519053518328, 21.62122265087433 53.824812272678315, 21.61966626688624 53.82462799104965, 21.61841771906319 53.824608425568755, 21.61571427548344 53.82394982966065, 21.61500807600177 53.82326878814874, 21.611872424197344 53.82165475795543, 21.609273987581474 53.82112110876264, 21.60839025161721 53.8207318880892, 21.60805589612579 53.820254304247584, 21.608127715248088 53.82008145747503, 21.607654620570287 53.819110102659465, 21.606991674823746 53.8185295797293, 21.606924339711952 53.81804840149357, 21.60666468596142 53.81755592345249, 21.606703357796476 53.81721346842346, 21.606256194768264 53.81705705852218, 21.606676060407466 53.816978782401975, 21.60649927487475 53.816838537320535, 21.606631864024735 53.816727645528715, 21.606808649556655 53.816704814829365, 21.606935714158027 53.816414537708994, 21.60658470840704 53.81639070619093, 21.606590232954773 53.816331998181, 21.60678359213071 53.81605150322039, 21.60668815504812 53.815268163595874, 21.607051314439445 53.81447354423318, 21.60697397076933 53.814261533000376, 21.6072214015347 53.81379234186258, 21.6072214015347 53.813365049643295, 21.607540606450556 53.813007222645496, 21.607546130998287 53.81285065513012, 21.607910751159267 53.81262558830261, 21.607872079215582 53.812471440765734, 21.607645572751522 53.81236053741864, 21.60792180014616 53.8121452536129, 21.608170923027814 53.811320254445064, 21.608579739572377 53.81055695156462, 21.60869275027477 53.80910916454505, 21.609034101960333 53.80814747004527, 21.609431869408553 53.80784408478243, 21.60957550636968 53.806757758528335, 21.610455584350774 53.805690059968754, 21.61059922259534 53.8054616922889, 21.612121328045276 53.804072899688805, 21.613231761587514 53.80358678287416, 21.614848784890256 53.80283255639833, 21.615318371460745 53.80239536452714, 21.615612322121564 53.802424388818, 21.615794632201244 53.802721288066806, 21.615954844090766 53.802793065591544, 21.61646310249637 53.802783277754315, 21.61666751076862 53.8028746308112, 21.616932689066914 53.80289094383619, 21.617313882871144 53.8026527730421, 21.61750723713601 53.802725701066066, 21.618065216472985 53.80263108506037, 21.618258575649037 53.802399438075724, 21.618755784959063 53.80226893216886, 21.61963596870757 53.802218269252194, 21.620387307220483 53.80230309824057, 21.62079059921652 53.80257389732341, 21.621386151139575 53.80273070530541, 21.6223474224725 53.80281227064634, 21.622651272606646 53.80279922020276, 21.62350205298145 53.80279269497899, 21.624104228701157 53.80270134174356, 21.624357977426996 53.80241120573302, 21.62395468543184 53.80187286640012, 21.62441322290607 53.80175867231938, 21.62510931594079 53.80213061760983, 21.62531372421219 53.80240794309128, 21.625774893789355 53.80256150430645, 21.626316299482227 53.80257455482399, 21.626940573393767 53.80248320111332, 21.62765324007168 53.80227765453668, 21.628262503109653 53.80203109269621, 21.628875727925703 53.80154821456995, 21.629549722767706 53.80136876520021, 21.63011659752638 53.800821896050934, 21.630569610452852 53.80068812245855, 21.63086241149159 53.80040752280513, 21.63156402907316 53.80031616437316, 21.631945222877363 53.80032921559007, 21.632481104023327 53.80056413679654, 21.633221393439925 53.80053477171796, 21.633585541686017 53.801050242901454, 21.634198766502067 53.80128842279811, 21.634707024907613 53.80141240631815, 21.63534982491484 53.801332265117, 21.635797313293637 53.801201755888655, 21.636283473508286 53.80112997563987, 21.6363969956964 53.80115266656827))"

aoi = loads(wkt_src)

# ---------------------------------------------
# Obliczenie bounding boxa z buforem
# ---------------------------------------------
inflate_bbox = 0.1
minx, miny, maxx, maxy = aoi.bounds
delx = maxx - minx
dely = maxy - miny
minx -= delx * inflate_bbox
maxx += delx * inflate_bbox
miny -= dely * inflate_bbox
maxy += dely * inflate_bbox

inflated_bbox = BBox(bbox=[minx, miny, maxx, maxy], crs=CRS.WGS84)
bbox_shape = box(minx, miny, maxx, maxy)

# ---------------------------------------------
# Konfiguracja dostępu do Sentinel Hub
# ---------------------------------------------
config = SHConfig()
config.sh_client_id = 'fa6a24b1-a9d6-4fbd-9884-de2af9e25bb2'
config.sh_client_secret = 'u5FTiCW6hqTBryWkxMIizzRj3zSOeU01'

# ---------------------------------------------
# Definicja zadań przetwarzania
# ---------------------------------------------
input_task = SentinelHubInputTask(
    data_collection=DataCollection.SENTINEL2_L1C,
    bands_feature=(FeatureType.DATA, 'BANDS'),
    bands=['B02', 'B03', 'B04', 'B08'],
    additional_data=[
        (FeatureType.MASK, 'CLM'),
        (FeatureType.MASK, 'dataMask'),
    ],
    resolution=10,
    maxcc=0.3,
    config=config
)

calculate_ndwi = NormalizedDifferenceIndexTask(
    (FeatureType.DATA, 'BANDS'),
    (FeatureType.DATA, 'NDWI'),
    (1, 3)  # B3 (zielony) i B8 (NIR) dla NDWI
)

lukajno_gdf = gpd.GeoDataFrame(crs=CRS.WGS84.pyproj_crs(), geometry=[aoi])

Add_Nominal_Water = VectorToRasterTask(
    vector_input=lukajno_gdf,
    raster_feature=(FeatureType.MASK_TIMELESS, "NOMINAL_WATER"),
    values=1,
    raster_shape=(FeatureType.MASK, "dataMask"),
    raster_dtype=np.uint8
)


class AddValidDataMaskTask(EOTask):
    def execute(self, eopatch):
        is_data_mask = eopatch[FeatureType.MASK, "dataMask"].astype(bool)
        cloud_mask = ~eopatch[FeatureType.MASK, "CLM"].astype(bool)
        eopatch[FeatureType.MASK, "VALID_DATA"] = np.logical_and(is_data_mask, cloud_mask)
        return eopatch


AddValidDataMask = AddValidDataMaskTask()


def calculate_coverage(array):
    return 1.0 - np.count_nonzero(array) / np.size(array)


class AddValidDataCoverageTask(EOTask):
    def execute(self, eopatch):
        valid_data = eopatch[FeatureType.MASK, "VALID_DATA"]
        time, height, width, channels = valid_data.shape
        coverage = np.apply_along_axis(calculate_coverage, 1, valid_data.reshape((time, height * width * channels)))
        eopatch[FeatureType.SCALAR, "COVERAGE"] = coverage[:, np.newaxis]
        return eopatch


add_coverage = AddValidDataCoverageTask()


class ValidDataCoveragePredicate:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, array):
        return calculate_coverage(array) < self.threshold


remove_cloudy_scenes = SimpleFilterTask(
    (FeatureType.MASK, "VALID_DATA"),
    ValidDataCoveragePredicate(0.05)
)


class WaterDetectionTask(EOTask):
    @staticmethod
    def detect_water(ndwi):
        otsu_thr = 1.0
        if len(np.unique(ndwi)) > 1:
            ndwi[np.isnan(ndwi)] = -1
            otsu_thr = threshold_otsu(ndwi)
        return ndwi > otsu_thr

    def execute(self, eopatch):
        water_masks = np.asarray([self.detect_water(ndwi[..., 0]) for ndwi in eopatch.data["NDWI"]])
        water_masks = water_masks[..., np.newaxis] * eopatch.mask_timeless["NOMINAL_WATER"]

        water_levels = np.asarray(
            [np.count_nonzero(mask) / np.count_nonzero(eopatch.mask_timeless["NOMINAL_WATER"])
             for mask in water_masks]
        )

        eopatch[FeatureType.MASK, "WATER_MASK"] = water_masks
        eopatch[FeatureType.SCALAR, "WATER_LEVEL"] = water_levels[..., np.newaxis]
        return eopatch


water_detection = WaterDetectionTask()

# ---------------------------------------------
# Definicja workflow i wykonanie
# ---------------------------------------------
output_task = OutputTask("final_eopatch")

workflow_nodes = linearly_connect_tasks(
    input_task,
    calculate_ndwi,
    Add_Nominal_Water,
    AddValidDataMask,
    add_coverage,
    remove_cloudy_scenes,
    water_detection,
    output_task,
)

workflow = EOWorkflow(workflow_nodes)

time_interval = ('2022-06-01', '2022-08-31')
result = workflow.execute({
    workflow_nodes[0]: {"bbox": inflated_bbox, "time_interval": time_interval},
})

eopatch = result.outputs["final_eopatch"]

# ---------------------------------------------
# Zbieranie danych z obieku 'eopatch'
# ---------------------------------------------

dates = [ts.date() for ts in eopatch.timestamp]
ndwi_mean = eopatch.data['NDWI'][..., 0].mean(axis=(1, 2))
water_level = eopatch.scalar['WATER_LEVEL'][..., 0].flatten()
day_of_year = [d.timetuple().tm_yday for d in dates]

# ---------------------------------------------
# Tworzenie DataFrame
# ---------------------------------------------
df = pd.DataFrame({
    'date': dates,
    'day_of_year': day_of_year,
    'ndwi_mean': ndwi_mean,
    'water_level': water_level
})

# ---------------------------------------------
# Sprawdzenie danych
# ---------------------------------------------
print(df)

# ---------------------------------------------
# Zapis danych do pliku CSV (opcjonalnie)
df.to_csv("dane_ml_lukajno.csv", index=False)
# ---------------------------------------------






# ---------------------------------------------
# ETAP 4 - AKTUALIZACJA W KODZIE

# Przygotowanie danych: 

# CECHY WEJŚĆIOWE 
X = df[['ndwi_mean', 'day_of_year']]  

# WARTOŚĆ DO PRZEWIDZENIA
y = df['water_level']                

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Trening modelu 
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predykcja i ocena 
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f"Średni błąd MAE: {mae:.4f}")

# Wizualizacja wyników 
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Rzeczywisty poziom wody', marker='o')
plt.plot(y_pred, label='Przewidywany poziom wody', marker='x')
plt.title("Porównanie: rzeczywisty vs przewidywany poziom wody")
plt.xlabel("Indeks próbki testowej")
plt.ylabel("Poziom wody (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------------





# ---------------------------------------------
# Wizualizacja wyników
# ---------------------------------------------
def plot_water_levels(eopatch, scene_idx=0):
    # Przygotowanie obrazu RGB
    rgb_raw = eopatch.data['BANDS'][scene_idx][..., [2, 1, 0]].astype(float)
    vmin, vmax = 0.02, 0.25
    rgb = np.clip((rgb_raw - vmin) / (vmax - vmin), 0, 1)

    # Pobranie masek wody
    water_mask = eopatch.mask['WATER_MASK'][scene_idx][..., 0]
    nominal_water_mask = eopatch.mask_timeless['NOMINAL_WATER'][..., 0]

    # Znajdowanie konturów
    def get_largest_contour(mask):
        contours = find_contours(mask, 0.5)
        return max(contours, key=len) if contours else None

    current_contour = get_largest_contour(water_mask)
    observable_contour = get_largest_contour(nominal_water_mask)

    # Przygotowanie wykresu
    fig, ax = plt.subplots(figsize=(12, 12))

    # Wyświetlenie obrazu RGB
    ax.imshow(rgb)

    # Dodanie konturów
    if current_contour is not None:
        ax.plot(current_contour[:, 1], current_contour[:, 0],
                linewidth=2, color='blue', label='Aktualny poziom wody')

    if observable_contour is not None:
        ax.plot(observable_contour[:, 1], observable_contour[:, 0],
                linewidth=2, color='red', linestyle='--', label='Obserwowalny poziom wody')

    # Dodanie informacji
    date = eopatch.timestamp[scene_idx].date()
    water_level = eopatch.scalar['WATER_LEVEL'][scene_idx][0] * 100
    ax.set_title(f"Jezioro Łukajno - {date}\nPoziom wody: {water_level:.1f}% obszaru nominalnego")
    ax.legend(loc='upper right')
    ax.axis('off')

    plt.tight_layout()
    plt.show()


# Wyświetlenie wyników dla pierwszej dostępnej sceny
if len(eopatch.timestamp) > 0:
    plot_water_levels(eopatch, 0)
else:
    print("Brak dostępnych scen spełniających kryteria (zachmurzenie < 5%)")