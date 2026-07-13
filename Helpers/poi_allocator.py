"""
Two-level zone→POI spatial allocator for SynTravelers.

Approach: inspired by ASTRA (Schneider et al., 2025) — two-level zone→POI allocation —
simplified because SynTravelers already has behavioral grounding from ODiN
(trip distances, purposes, timing are empirically derived). The spatial layer
just places those trips at real GPS locations.

Thesis citation:
    "A two-level zone→POI spatial allocation approach inspired by ASTRA
    (Schneider et al., 2025), adapted for survey-grounded generation by
    substituting ODiN empirical distance distributions for the EPR model and
    replacing SBERT semantic mapping with a purpose-to-tag lookup table."

Key design decisions
────────────────────
1. No EPR model — trip structure comes from ODiN behavioral grounding; EPR not needed.
2. No SBERT — purpose→tag lookup substitutes for semantic embedding; defensible
   because ODiN purposes are well-defined categorical labels.
3. S_frq retained — POI cluster density weighting from ASTRA improves spatial
   realism (office parks, shopping centres preferred over isolated POIs).
4. Distance bounds as proxy for PT — ODiN bounds applied with 3× soft filter;
   future work should recalibrate using IMOB 2017 trip-distance microdata when
   available. Portuguese work commutes average 14.8 km (IMOB 2017 AML) vs
   shorter Dutch cycling-dominated commutes, so the soft filter is important.
5. Visiting friends fallback — residential zone centroid + random offset; no OSM
   equivalent for private homes.
6. Zone unit for PT: 1km² uniform population grid (GRID1K21_CONT.gpkg, EPSG:3035)
   rather than named BGRI statistical zones. Uniform cell size simplifies
   density weighting and spatial joins; population column is N_INDIVIDUOS.
"""

import os
import warnings
from math import radians, cos, sin, asin, sqrt

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box as shapely_box


# ── ODiN distance class bounds ────────────────────────────────────────────────
# Full set of distance class labels as output by the trajectory generation
# pipeline (matches spatial_assignment.py DISTANCE_CLASS_RANGE exactly).
DISTANCE_CLASS_BOUNDS: dict[str, tuple[float, float]] = {
    "0.1 to 0.5 km":   (0.1,    0.5),
    "0.5 to 1.0 km":   (0.5,    1.0),
    "1.0 to 2.5 km":   (1.0,    2.5),
    "2.5 to 3.7 km":   (2.5,    3.7),
    "3.7 to 5.0 km":   (3.7,    5.0),
    "5.0 to 7.5 km":   (5.0,    7.5),
    "7.5 to 10 km":    (7.5,   10.0),
    "10 to 15 km":     (10.0,  15.0),
    "15 to 20 km":     (15.0,  20.0),
    "20 to 30 km":     (20.0,  30.0),
    "30 to 40 km":     (30.0,  40.0),
    "40 to 50 km":     (40.0,  50.0),
    "50 to 75 km":     (50.0,  75.0),
    "75 to 100 km":    (75.0, 100.0),
    "100 km or more":  (100.0, 300.0),
    "_default":        (1.0,   25.0),   # fallback for unrecognised labels
}

# ── Purpose → OSM tag mapping ─────────────────────────────────────────────────
# Keys are the exact purpose labels produced by the ODiN-grounded generation
# pipeline (trajectory_generation.py / trajectory_generation_portugal.py).
# Each value is an ordered list of tag dicts; the allocator tries tags in order
# and falls back to the next if no matches are found in the zone.
PURPOSE_TO_OSM_TAGS: dict[str, list[dict]] = {
    "To and from work": [
        {"office": True},
        {"amenity": "workplace"},
        {"building": "office"},
        {"building": "commercial"},
        {"landuse": "commercial"},
        {"landuse": "industrial"},
    ],
    "Shopping/grocery shopping": [
        {"shop": "supermarket"},
        {"shop": "convenience"},
        {"shop": "mall"},
        {"shop": "department_store"},
        {"shop": True},
        {"building": "retail"},
        {"building": "commercial"},
        {"landuse": "retail"},
    ],
    "Taking education/course": [
        {"amenity": "school"},
        {"amenity": "university"},
        {"amenity": "college"},
        {"amenity": "kindergarten"},
        {"building": "school"},
        {"building": "university"},
    ],
    "Sports/hobbies": [
        {"leisure": "sports_centre"},
        {"leisure": "fitness_centre"},
        {"leisure": "pitch"},
        {"leisure": "park"},
        {"leisure": "stadium"},
    ],
    "Services/personal care": [
        {"amenity": "pharmacy"},
        {"amenity": "bank"},
        {"amenity": "hospital"},
        {"amenity": "clinic"},
        {"amenity": "post_office"},
        {"building": "hospital"},
        {"building": "public"},
    ],
    # Both social visit variants map to residential centroid fallback
    "Social or family visit": [
        {"landuse": "residential"},
        {"building": "residential"},
        {"building": "apartments"},
        {"building": "house"},
    ],
    "Visitors/staying over": [
        {"landuse": "residential"},
        {"building": "residential"},
        {"building": "apartments"},
        {"building": "house"},
    ],
    "Other leisure activities": [
        {"leisure": "park"},
        {"tourism": "attraction"},
        {"amenity": "restaurant"},
        {"amenity": "cafe"},
        {"amenity": "bar"},
    ],
    "Touring/hiking": [
        {"leisure": "park"},
        {"tourism": "attraction"},
        {"natural": "peak"},
    ],
    "Pick up/drop off people": [
        {"amenity": "school"},
        {"building": "school"},
        {"amenity": "bus_station"},
        {"public_transport": "stop_position"},
        {"landuse": "residential"},
    ],
    "Different motive": [
        {"amenity": True},
    ],
    "_fallback": [
        {"amenity": True},
    ],
}

# ── Destination → OSM tag mapping ─────────────────────────────────────────────
# Keys match the controlled DESTINATION_TYPES vocabulary in trajectory_generation.py.
# Ordered by specificity; allocate() tries tags in list order and falls back
# to PURPOSE_TO_OSM_TAGS when no destination-matched POI is found in the zone.
# "home" and "other" are absent: "home" triggers an early-return in allocate();
# "other" signals an unresolvable destination, so allocate() uses purpose tags.
DESTINATION_TO_OSM_TAGS: dict[str, list[dict]] = {
    "workplace": [
        {"office": True},
        {"amenity": "workplace"},
        {"building": "office"},
        {"building": "commercial"},
        {"landuse": "commercial"},
        {"landuse": "industrial"},
    ],
    "educational institution": [
        {"amenity": "school"},
        {"amenity": "university"},
        {"amenity": "college"},
        {"amenity": "kindergarten"},
        {"building": "school"},
        {"building": "university"},
    ],
    "supermarket or shop": [
        {"shop": "supermarket"},
        {"shop": "convenience"},
        {"shop": "mall"},
        {"shop": "department_store"},
        {"shop": True},
        {"building": "retail"},
        {"building": "commercial"},
        {"landuse": "retail"},
    ],
    "sports or recreation facility": [
        {"leisure": "sports_centre"},
        {"leisure": "fitness_centre"},
        {"leisure": "pitch"},
        {"leisure": "stadium"},
        {"leisure": "park"},
    ],
    "social or family visit destination": [
        {"landuse": "residential"},
        {"building": "residential"},
        {"building": "apartments"},
        {"building": "house"},
    ],
    "healthcare or personal service": [
        {"amenity": "pharmacy"},
        {"amenity": "hospital"},
        {"amenity": "clinic"},
        {"amenity": "bank"},
        {"amenity": "post_office"},
        {"building": "hospital"},
    ],
    "park or nature area": [
        {"leisure": "park"},
        {"natural": "wood"},
        {"leisure": "nature_reserve"},
        {"landuse": "forest"},
    ],
    "restaurant or café": [
        {"amenity": "restaurant"},
        {"amenity": "cafe"},
        {"amenity": "bar"},
        {"amenity": "fast_food"},
    ],
    "transit hub": [
        {"public_transport": "station"},
        {"amenity": "bus_station"},
        {"railway": "station"},
        {"public_transport": "stop_position"},
    ],
}

# Residential-visit purposes that use the centroid fallback
_RESIDENTIAL_PURPOSES = frozenset({"Social or family visit", "Visitors/staying over"})

# Destination labels that also map to residential centroid (no OSM tag for private homes)
_RESIDENTIAL_DESTINATIONS = frozenset({"social or family visit destination"})

# Ordered keys for fuzzy purpose normalization
_PURPOSE_KEYS = [k for k in PURPOSE_TO_OSM_TAGS if not k.startswith("_")]

# Ordered keys for fuzzy destination normalization
_DESTINATION_KEYS = list(DESTINATION_TO_OSM_TAGS.keys())

# Tags requested when building the OSMnx POI cache.
# Excludes street furniture / infrastructure (parking, bench, vending_machine, etc.)
# that bloats the cache and wastes zone search budget without adding useful destinations.
# "building" added to catch PT/Southern-European cities where workplaces, schools,
# and shops are often tagged at the building level rather than with amenity/shop/office.
_CACHE_TAGS = {
    "amenity": [
        "workplace", "school", "university", "college", "kindergarten",
        "pharmacy", "bank", "hospital", "clinic", "post_office",
        "restaurant", "cafe", "bar", "fast_food", "bus_station",
    ],
    "shop": ["supermarket", "convenience", "mall", "department_store"],
    "leisure": ["sports_centre", "fitness_centre", "pitch", "park", "stadium", "nature_reserve"],
    "office": True,
    "tourism": ["attraction"],
    "landuse": ["residential", "commercial", "industrial", "retail", "forest"],
    "building": ["office", "commercial", "retail", "school", "university",
                 "hospital", "public", "residential", "apartments", "house"],
    "natural": ["wood", "peak"],
    "public_transport": ["station", "stop_position"],
    "railway": ["station"],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_purpose(purpose: str) -> str:
    """
    Map a raw generation-pipeline purpose string to a canonical PURPOSE_TO_OSM_TAGS key.

    Some generated trips contain long malformed strings where the LLM was
    uncertain about the correct purpose category. These are caught by substring
    matching; anything unresolved falls back to "_fallback" (any amenity).
    """
    if not isinstance(purpose, str):
        return "_fallback"
    p = purpose.strip()
    if p in PURPOSE_TO_OSM_TAGS:
        return p
    p_lower = p.lower()
    for key in _PURPOSE_KEYS:
        if key.lower() in p_lower:
            return key
    return "_fallback"


def _normalize_destination(destination: str) -> str | None:
    """
    Map a destination string to a DESTINATION_TO_OSM_TAGS key.

    Returns None for "other", "home", empty strings, or non-string input so
    that allocate() knows to fall back to purpose-based tags without an
    explicit branch for each unrecognised value.
    """
    if not isinstance(destination, str):
        return None
    d = destination.strip().lower()
    if d in ("other", "home", ""):
        return None
    for key in _DESTINATION_KEYS:
        if d == key.lower():
            return key
        if key.lower() in d:
            return key
    return None


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Scalar haversine distance in km."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(max(a, 0.0)))


def _haversine_vec(
    home_lat: float,
    home_lon: float,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """Vectorised haversine from one point to an array of points (km)."""
    R = 6371.0
    dlat = np.radians(lats - home_lat)
    dlon = np.radians(lons - home_lon)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(radians(home_lat)) * np.cos(np.radians(lats)) * np.sin(dlon / 2) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def compute_s_frq(candidate_pois: gpd.GeoDataFrame, radius_m: float = 500) -> pd.Series:
    """
    S_frq: for each POI in candidate_pois, count how many other candidates share
    a cluster within radius_m.  Returns a normalized weight Series (sum=1).

    Uses a spatial index over candidates (not the full POI cache) — sufficient
    because candidates are already filtered to a single zone and purpose tag,
    so intra-cluster density is captured.
    """
    n = len(candidate_pois)
    if n == 0:
        return pd.Series(dtype=float)
    if n == 1:
        return pd.Series([1.0], index=candidate_pois.index)

    pois_m = candidate_pois.to_crs("EPSG:3857")
    sindex = pois_m.sindex
    geoms = pois_m.geometry.values

    counts = np.zeros(n, dtype=float)
    for i, geom in enumerate(geoms):
        buf = geom.buffer(radius_m)
        hits = list(sindex.intersection(buf.bounds))
        # Count strict intersections minus self
        counts[i] = sum(1 for j in hits if j != i and geoms[j].intersects(buf))

    counts += 1.0  # Laplace smoothing: isolated POIs still get weight 1
    return pd.Series(counts / counts.sum(), index=candidate_pois.index)


# ── Main class ────────────────────────────────────────────────────────────────

class POIAllocator:
    """
    Two-level zone→POI spatial allocator for SynTravelers.

    Level 1 — Zone selection
        Given a trip purpose and distance class, sample a destination zone
        weighted by population density within the ODiN distance band.

    Level 2 — POI selection within zone
        Filter the POI cache to the selected zone and matching purpose tags.
        Apply S_frq density weighting; sample from top-k=5 candidates.
        Soft distance validation: resample up to MAX_RESAMPLE_ATTEMPTS if the
        selected POI is > 3× the upper distance bound from agent home.

    Parameters
    ----------
    country : "NL" | "PT"
    poi_cache_path : path to .gpkg POI cache (built once by build_poi_cache)
    zone_data_path : zone boundary file
        PT → Portugal Data/GRID1K21_CONT.gpkg  (1km² population grid)
        NL → CBS PC4 shapefile with population column
    study_area_boundary : shapely Polygon in WGS84 to restrict zone candidates.
        PT: if None, derived from the POI cache bounding box (±0.05°).
        NL: if None, all PC4 zones are used.
    seed : int for reproducible sampling
    """

    TOP_K_POIS = 5
    MAX_RESAMPLE_ATTEMPTS = 3
    S_FRQ_RADIUS_M = 500

    def __init__(
        self,
        country: str,
        poi_cache_path: str,
        zone_data_path: str,
        study_area_boundary=None,
        seed: int = 42,
    ):
        if country not in ("NL", "PT"):
            raise ValueError("country must be 'NL' or 'PT'")
        self.country = country
        self._rng = np.random.default_rng(seed)

        # ── Load and prepare POI cache ────────────────────────────────────────
        self._pois: gpd.GeoDataFrame = gpd.read_file(poi_cache_path).to_crs("EPSG:4326")
        # OSMnx returns polygon/linestring features too; reduce to centroids for allocation
        mask_non_point = self._pois.geometry.geom_type != "Point"
        if mask_non_point.any():
            self._pois = self._pois.copy()
            metric = self._pois.loc[mask_non_point].to_crs("EPSG:3857")
            centroids_wgs84 = gpd.GeoSeries(metric.geometry.centroid, crs="EPSG:3857").to_crs("EPSG:4326")
            self._pois.loc[mask_non_point, "geometry"] = centroids_wgs84.values
        # Exclude amenity values that are noise: infrastructure/street furniture
        # that was pulled into the cache via other matching tags (e.g. office=*)
        # but is never a meaningful trip destination.
        _AMENITY_NOISE = {
            "parking", "parking_entrance", "parking_space",
            "bench", "vending_machine", "charging_station",
            "recycling", "fountain", "drinking_water",
            "shelter", "waste_basket", "waste_disposal",
            "bicycle_parking", "motorcycle_parking",
            "compressed_air", "telephone",
        }
        _LEISURE_BLACKLIST = {"swimming_pool", "garden"}
        if "amenity" in self._pois.columns:
            noise_mask = self._pois["amenity"].isin(_AMENITY_NOISE)
            self._pois = self._pois[~noise_mask]
        if "leisure" in self._pois.columns:
            blacklist_mask = self._pois["leisure"].isin(_LEISURE_BLACKLIST)
            self._pois = self._pois[~blacklist_mask]

        self._pois = (
            self._pois[~self._pois.geometry.is_empty & self._pois.geometry.notna()]
            .reset_index(drop=True)
        )

        # ── Load zone data ────────────────────────────────────────────────────
        if country == "PT":
            self._zones, self._zone_id_col, self._zone_pop_col = self._load_pt_zones(
                zone_data_path, study_area_boundary
            )
        else:
            self._zones, self._zone_id_col, self._zone_pop_col = self._load_nl_zones(
                zone_data_path, study_area_boundary
            )

        # Pre-compute zone centroids and population arrays for vectorised zone selection
        # Reproject to metric CRS for accurate centroid, then back to WGS84 for lat/lon
        centroids = gpd.GeoSeries(
            self._zones.to_crs("EPSG:3857").geometry.centroid, crs="EPSG:3857"
        ).to_crs("EPSG:4326")
        self._zone_lats = np.asarray(centroids.y, dtype=float)
        self._zone_lons = np.asarray(centroids.x, dtype=float)
        self._zone_pops = np.asarray(
            self._zones[self._zone_pop_col].fillna(0).clip(lower=0), dtype=float
        )

        print(
            f"POIAllocator ({country}): {len(self._pois)} POIs, "
            f"{len(self._zones)} zones loaded."
        )

    # ── Zone loaders ──────────────────────────────────────────────────────────

    def _load_pt_zones(self, zone_data_path, study_area_boundary):
        """
        Load the 1km² population grid for Portugal and filter to the study area.

        The file GRID1K21_CONT.gpkg covers all of continental Portugal (93k cells).
        Filtering is required; without it zone selection would sample destinations
        across the entire country.

        If study_area_boundary is not provided, the bounding box of the POI cache
        (± 0.05° ≈ 5km margin) is used as a proxy for the Oeiras study area.
        """
        zones_raw = gpd.read_file(zone_data_path).to_crs("EPSG:4326")
        zones_raw["_pop"] = pd.to_numeric(zones_raw["N_INDIVIDUOS"], errors="coerce").fillna(0)
        zones_raw["_zone_id"] = zones_raw.index.astype(str)

        if study_area_boundary is not None:
            # Centroid-within filter: keeps only cells whose centroid is inside the
            # boundary, avoiding edge cells whose centroid (and thus sampled home
            # coordinates) would fall in a neighbouring municipality.
            centroids = zones_raw.to_crs("EPSG:3857").geometry.centroid.to_crs("EPSG:4326")
            mask = centroids.within(study_area_boundary)
        else:
            minx, miny, maxx, maxy = self._pois.total_bounds
            study_bbox = shapely_box(minx - 0.05, miny - 0.05, maxx + 0.05, maxy + 0.05)
            mask = zones_raw.geometry.intersects(study_bbox)
            warnings.warn(
                f"PT: study_area_boundary not provided. Filtered to {mask.sum()} grid cells "
                f"using POI bounding box ±0.05°. Pass the Oeiras municipality polygon via "
                f"study_area_boundary for exact scope.",
                UserWarning,
                stacklevel=3,
            )

        zones = zones_raw[mask].copy().reset_index(drop=True)
        if len(zones) == 0:
            raise RuntimeError(
                "No PT grid cells survived the study area filter. "
                "Check that poi_cache_path and zone_data_path cover the same area."
            )
        return zones, "_zone_id", "_pop"

    def _load_nl_zones(self, zone_data_path, study_area_boundary=None):
        """
        Load PC4 postal code zones for NL. Accepts CBS PC4 shapefiles with varied
        column naming conventions; normalises to '_zone_id' and '_pop'.

        If study_area_boundary (shapely Polygon, WGS84) is provided, only zones
        whose centroid falls inside the boundary are retained — equivalent to the
        PT Oeiras filter. Pass a Rotterdam (or other city) polygon to scope the
        allocator to that city.
        """
        zones_raw = gpd.read_file(zone_data_path).to_crs("EPSG:4326")

        id_candidates = ("pc4", "postcode4", "pc_4", "postcode")
        pop_candidates = ("population", "pop", "inwoners", "aantal_inw", "bev_dichth")

        id_col = next(
            (c for c in zones_raw.columns if c.lower() in id_candidates), None
        )
        pop_col = next(
            (c for c in zones_raw.columns if c.lower() in pop_candidates), None
        )

        rename = {}
        if id_col:
            rename[id_col] = "_zone_id"
        if pop_col:
            rename[pop_col] = "_pop"
        zones_raw = zones_raw.rename(columns=rename)

        if "_zone_id" not in zones_raw.columns:
            zones_raw["_zone_id"] = zones_raw.index.astype(str)
        if "_pop" not in zones_raw.columns:
            warnings.warn(
                "NL: no population column found in zone file; using uniform weights.",
                UserWarning,
                stacklevel=3,
            )
            zones_raw["_pop"] = 1.0

        if study_area_boundary is not None:
            centroids = (
                zones_raw.to_crs("EPSG:28992").geometry.centroid
                .to_crs("EPSG:4326")
            )
            mask = centroids.within(study_area_boundary)
            zones_raw = zones_raw[mask].copy().reset_index(drop=True)
            if len(zones_raw) == 0:
                raise RuntimeError(
                    "NL: no zones survived the study area filter. "
                    "Check that study_area_boundary overlaps zone_data_path coverage."
                )

        return zones_raw.copy().reset_index(drop=True), "_zone_id", "_pop"

    # ── Public interface ──────────────────────────────────────────────────────

    def allocate(
        self,
        trip: dict,
        agent_home: tuple,
        origin: tuple | None = None,
    ) -> dict:
        """
        Assign GPS coordinates to a single trip.

        Parameters
        ----------
        trip       : dict — must contain 'purpose' and 'distance_class'.
                     If trip['destination'] == 'home', agent_home coordinates
                     are returned directly (no zone/POI lookup needed).
        agent_home : (lat, lon) — the agent's home address. Used only for the
                     home-return special case.
        origin     : (lat, lon) — spatial origin for zone selection and distance
                     validation; defaults to agent_home when None. Pass the
                     previous trip's destination to chain trips correctly.

        Returns
        -------
        Copy of trip enriched with:
            destination_lat, destination_lon,
            destination_poi_label, destination_zone_id
        """
        trip = dict(trip)
        home_lat, home_lon = agent_home
        orig_lat, orig_lon = origin if origin is not None else agent_home

        # Return-home trips: destination is always the agent's own home
        if str(trip.get("destination", "")).strip().lower() == "home":
            trip.update({
                "destination_lat": home_lat,
                "destination_lon": home_lon,
                "destination_poi_label": "home",
                "destination_zone_id": "home",
            })
            return trip

        purpose_raw = trip.get("purpose", "")
        distance_class = trip.get("distance_class", "_default")
        purpose_key = _normalize_purpose(purpose_raw)
        purpose_tags = PURPOSE_TO_OSM_TAGS.get(purpose_key, PURPOSE_TO_OSM_TAGS["_fallback"])

        # Destination tags take priority; purpose tags serve as fallback when
        # destination is "other", unrecognised, or matches no POI in the zone.
        dest_key = _normalize_destination(str(trip.get("destination", "")))
        if dest_key is not None:
            osm_tags = DESTINATION_TO_OSM_TAGS[dest_key]
            fallback_tags = purpose_tags
        else:
            osm_tags = purpose_tags
            fallback_tags = None

        lo_km, hi_km = DISTANCE_CLASS_BOUNDS.get(
            distance_class, DISTANCE_CLASS_BOUNDS["_default"]
        )

        is_residential = (
            purpose_key in _RESIDENTIAL_PURPOSES
            or dest_key in _RESIDENTIAL_DESTINATIONS
        )

        best_result: dict | None = None

        for _ in range(self.MAX_RESAMPLE_ATTEMPTS):
            zone_idx = self._select_zone(orig_lat, orig_lon, lo_km, hi_km)
            if zone_idx is None:
                continue

            zone_row = self._zones.iloc[zone_idx]
            zone_id = zone_row[self._zone_id_col]
            zone_geom = zone_row.geometry

            # Visiting friends/family — use residential zone centroid + random offset
            if is_residential:
                lat, lon = self._residential_centroid_fallback(zone_geom)
                result = None
            else:
                result = self._select_poi_in_zone(
                    zone_geom, zone_id, osm_tags, fallback_tags=fallback_tags
                )
                if result is None:
                    continue
                lat = result["destination_lat"]
                lon = result["destination_lon"]

            candidate = {
                "destination_lat": lat,
                "destination_lon": lon,
                "destination_poi_label": (
                    "residential_centroid"
                    if is_residential
                    else (result or {}).get("destination_poi_label", "unknown")
                ),
                "destination_zone_id": str(zone_id),
            }
            best_result = best_result or candidate

            if self._validate_distance(orig_lat, orig_lon, lat, lon, hi_km):
                trip.update(candidate)
                return trip

        # Accept best candidate even if soft distance filter was never satisfied
        if best_result:
            trip.update(best_result)
            return trip

        # Last resort: expanded zone search → zone centroid
        zone_idx = self._select_zone(orig_lat, orig_lon, lo_km, hi_km, expand=True)
        if zone_idx is not None:
            zone_row = self._zones.iloc[zone_idx]
            c = zone_row.geometry.centroid
            trip.update({
                "destination_lat": c.y,
                "destination_lon": c.x,
                "destination_poi_label": "zone_centroid_fallback",
                "destination_zone_id": str(zone_row[self._zone_id_col]),
            })
        return trip

    # ── Level 1: zone selection ───────────────────────────────────────────────

    def _select_zone(
        self,
        home_lat: float,
        home_lon: float,
        lo_km: float,
        hi_km: float,
        expand: bool = False,
    ) -> int | None:
        """
        Sample a destination zone index weighted by population within distance band.
        With expand=True, relaxes bounds by ±50% when no zone falls in range.
        """
        dists = _haversine_vec(home_lat, home_lon, self._zone_lats, self._zone_lons)
        mask = (dists >= lo_km) & (dists <= hi_km)

        if not mask.any():
            if expand:
                mask = (dists >= lo_km * 0.5) & (dists <= hi_km * 1.5)
            if not mask.any():
                return None

        pops = self._zone_pops[mask].clip(min=1.0)
        weights = pops / pops.sum()
        candidates = np.where(mask)[0]
        return int(self._rng.choice(candidates, p=weights))

    # ── Level 2: POI selection within zone ────────────────────────────────────

    def _select_poi_in_zone(
        self,
        zone_geom,
        zone_id,
        osm_tags: list[dict],
        fallback_tags: list[dict] | None = None,
    ) -> dict | None:
        """
        Select a POI from the cache within zone_geom matching osm_tags.
        If no osm_tags match and fallback_tags is provided, tries fallback_tags
        before accepting any POI in the zone.
        """
        # Spatial filter: POIs within the zone polygon
        zone_pois = self._pois[self._pois.geometry.within(zone_geom)]

        if zone_pois.empty:
            # Expand by ~1km if zone polygon contains no indexed POIs
            zone_pois = self._pois[self._pois.geometry.within(zone_geom.buffer(0.01))]

        if zone_pois.empty:
            return None

        # Tag filter: destination tags first; purpose fallback_tags if no match;
        # finally accept any POI in zone rather than returning nothing.
        # Track which tag set matched so _poi_label can use it as fallback label.
        matched_tags_used = None
        matched = self._filter_by_tags(zone_pois, osm_tags)
        if not matched.empty:
            matched_tags_used = osm_tags
        elif fallback_tags is not None:
            matched = self._filter_by_tags(zone_pois, fallback_tags)
            if not matched.empty:
                matched_tags_used = fallback_tags
        if matched.empty:
            matched = zone_pois
            matched_tags_used = osm_tags   # best available hint

        # S_frq density weighting; sample top-k candidates
        weights = compute_s_frq(matched, radius_m=self.S_FRQ_RADIUS_M)
        top_k_idx = weights.nlargest(min(self.TOP_K_POIS, len(weights))).index
        top_weights = weights.loc[top_k_idx]
        top_weights = top_weights / top_weights.sum()

        chosen_idx = self._rng.choice(top_k_idx, p=top_weights.values)
        chosen = matched.loc[chosen_idx]

        return {
            "destination_lat": float(chosen.geometry.y),
            "destination_lon": float(chosen.geometry.x),
            "destination_poi_label": self._poi_label(chosen, search_tags=matched_tags_used),
            "destination_zone_id": str(zone_id),
        }

    @staticmethod
    def _filter_by_tags(
        pois: gpd.GeoDataFrame, tags: list[dict]
    ) -> gpd.GeoDataFrame:
        """Return rows of pois matching at least one tag dict."""
        combined_mask = pd.Series(False, index=pois.index)
        for tag_dict in tags:
            tag_mask = pd.Series(True, index=pois.index)
            for k, v in tag_dict.items():
                if k not in pois.columns:
                    tag_mask = pd.Series(False, index=pois.index)
                    break
                col_str = pois[k].astype(str).str.strip().str.lower()
                if v is True:
                    # Any present (non-null) value
                    tag_mask &= pois[k].notna() & ~col_str.isin({"nan", "none", ""})
                else:
                    tag_mask &= col_str == str(v).lower()
            combined_mask |= tag_mask
        return pois[combined_mask]

    @staticmethod
    def _poi_label(poi_row, search_tags: list[dict] | None = None) -> str:
        """Return the OSM functional tag type for a POI row.

        Priority:
          1. Functional tag column on the POI (amenity, shop, leisure, office, …)
          2. The search tag that was used to find this POI — guarantees a
             classifiable label even when the POI has only a name in OSM
             (e.g. a company building found via {"office": True} → "office")
          3. The POI name as last resort
        """
        for col in ("amenity", "shop", "leisure", "office", "tourism", "landuse"):
            val = poi_row.get(col)
            if pd.notna(val) and str(val).strip().lower() not in ("nan", "none", "true", ""):
                return str(val).strip()
        # Use the search tag value as a classifiable fallback
        if search_tags:
            for tag_dict in search_tags:
                for k, v in tag_dict.items():
                    if isinstance(v, str) and v.lower() not in ("nan", "none", "true", ""):
                        return v          # e.g. "supermarket", "school", "company"
                    if v is True:
                        return k          # e.g. "office", "shop"
        # Last resort: POI name
        name = poi_row.get("name")
        if pd.notna(name) and str(name).strip().lower() not in ("nan", "none", ""):
            return str(name).strip()
        return "unknown"

    # ── Visiting friends/family fallback ──────────────────────────────────────

    def _residential_centroid_fallback(
        self, zone_geometry, offset_m: float = 100
    ) -> tuple[float, float]:
        """
        Returns zone centroid with a small random offset to simulate a home address.
        Used for 'Social or family visit' / 'Visitors/staying over' — OSM has no
        tag for private residences, o zone centroid is the best available proxy.
        """
        c = zone_geometry.centroid
        deg_offset = offset_m / 111_000
        lat = c.y + self._rng.uniform(-deg_offset, deg_offset)
        lon = c.x + self._rng.uniform(-deg_offset, deg_offset)
        return float(lat), float(lon)

    # ── Distance validation ────────────────────────────────────────────────────

    @staticmethod
    def _validate_distance(
        home_lat: float,
        home_lon: float,
        poi_lat: float,
        poi_lon: float,
        upper_km: float,
    ) -> bool:
        """
        Soft distance filter: accept if actual distance ≤ 3× upper bound.

        Distance bounds derived from ODiN (NL); applied as plausibility proxy for PT
        pending Portuguese trip-distance microdata. The 3× multiplier is more
        permissive for PT because Dutch short-distance classes (<5km) are dominated
        by cycling and underestimate car-based PT commute distances.
        """
        return _haversine(home_lat, home_lon, poi_lat, poi_lon) <= upper_km * 3.0

    # ── Home coordinate sampling ──────────────────────────────────────────────

    def sample_agent_home(self, offset_m: float = 400) -> tuple[float, float]:
        """
        Sample a home coordinate from population-weighted zone centroids.

        PT: weighted by N_INDIVIDUOS across Oeiras 1km² grid cells.
        NL: weighted by population across PC4 zones.

        A random offset (default ±400m) is added so agents in the same
        dominant zone don't all collapse to the same centroid.
        """
        pops = self._zone_pops.clip(min=1.0)
        weights = pops / pops.sum()
        zone_idx = int(self._rng.choice(len(self._zones), p=weights))
        c = self._zones.iloc[zone_idx].geometry.centroid
        deg_offset = offset_m / 111_000
        lat = float(c.y + self._rng.uniform(-deg_offset, deg_offset))
        lon = float(c.x + self._rng.uniform(-deg_offset, deg_offset))
        return lat, lon

    # ── Static: build POI cache ────────────────────────────────────────────────

    @staticmethod
    def build_poi_cache(place_query: str, output_path: str) -> None:
        """
        Fetch POIs from OpenStreetMap via OSMnx and write to a GeoPackage.

        Run once per study area; reuse the .gpkg for all subsequent allocations.

        For Rotterdam:
            POIAllocator.build_poi_cache("Rotterdam, Netherlands", "nl_rotterdam_pois.gpkg")

        Note on NL: querying "Netherlands" downloads millions of features and can
        take several hours. Scope to a city name for practical run times.
        """
        import osmnx as ox
        ox.settings.overpass_url = "https://overpass-api.de/api/interpreter"
        ox.settings.overpass_settings = "[out:json][timeout:600]"
        ox.settings.requests_timeout = 660
        print(f"  Fetching POIs for '{place_query}' …")
        # features_from_place uses the actual admin boundary polygon (much smaller
        # than the bbox, which inflates the query area to ~9500× for cities like Rotterdam).
        pois = ox.features_from_place(place_query, tags=_CACHE_TAGS)
        print(f"  {len(pois)} features retrieved.")
        pois.to_file(output_path, driver="GPKG")
        print(f"  Written → {output_path}")

    @staticmethod
    def build_nl_grid(
        boundary,
        output_path: str,
        cell_size_km: float = 1.0,
    ) -> None:
        """
        Build a uniform grid of square cells over a study-area boundary and write
        to GeoPackage. Intended for NL cities where CBS PC4 zones are unavailable
        or too coarse. All cells receive _pop = 1.0 (uniform weight); join CBS
        population data afterwards if available.

        Parameters
        ----------
        boundary    : shapely Polygon in WGS84 (EPSG:4326) defining the city boundary.
                      For Rotterdam:
                          import osmnx as ox
                          boundary = ox.geocode_to_gdf("Rotterdam, Netherlands").geometry.iloc[0]
        output_path : path for the output .gpkg (e.g. "nl_rotterdam_zones.gpkg").
        cell_size_km: grid resolution in km (default 1.0 → 1 km² cells, matching PT grid).
        """
        from shapely.geometry import box as _box

        boundary_gdf = gpd.GeoDataFrame(
            geometry=[boundary], crs="EPSG:4326"
        ).to_crs("EPSG:28992")
        minx, miny, maxx, maxy = boundary_gdf.total_bounds

        cell_m = cell_size_km * 1000.0
        xs = np.arange(minx, maxx, cell_m)
        ys = np.arange(miny, maxy, cell_m)

        bdry_proj = boundary_gdf.geometry.iloc[0]
        cells = [
            _box(x, y, x + cell_m, y + cell_m)
            for x in xs
            for y in ys
            if bdry_proj.intersects(_box(x, y, x + cell_m, y + cell_m))
        ]

        grid = gpd.GeoDataFrame(geometry=cells, crs="EPSG:28992").to_crs("EPSG:4326")
        grid["_zone_id"] = grid.index.astype(str)
        grid["_pop"] = 1.0
        grid.to_file(output_path, driver="GPKG")
        print(f"NL grid: {len(grid)} cells ({cell_size_km} km²) written → {output_path}")


# ── Convenience: allocate an entire trajectory ────────────────────────────────

def allocate_trajectory(
    record: dict,
    allocator: POIAllocator,
    agent_home: tuple,
) -> dict:
    """
    Apply POIAllocator.allocate to every trip in a trajectory record, chaining
    the destination of each trip as the origin of the next.

    Parameters
    ----------
    record     : dict with a 'trips' list (as produced by trajectory_generation*.py)
    allocator  : a constructed POIAllocator instance
    agent_home : (lat, lon) — sampled or fixed home coordinates for this agent

    Returns
    -------
    record with each trip enriched with destination_lat/lon/poi_label/zone_id;
    also adds agent_home_lat / agent_home_lon at the record level.
    """
    record = dict(record)
    record["agent_home_lat"] = agent_home[0]
    record["agent_home_lon"] = agent_home[1]

    current_origin = agent_home
    enriched_trips = []
    for trip in record.get("trips", []):
        enriched = allocator.allocate(trip, agent_home, origin=current_origin)
        dest_lat = enriched.get("destination_lat")
        dest_lon = enriched.get("destination_lon")
        if dest_lat is not None and dest_lon is not None:
            current_origin = (dest_lat, dest_lon)
        enriched_trips.append(enriched)

    record["trips"] = enriched_trips
    return record


# ── CLI: build POI caches ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building PT POI cache for Oeiras …")
    POIAllocator.build_poi_cache("Oeiras, Portugal", "oeiras_pois.gpkg")

    print()
    # Querying all of "Netherlands" is impractical (~hours, multi-GB file).
    # Default to Utrecht as a representative urban study area for NL.
    # Change the place_query below to match your actual NL study area.
    print("Building NL POI cache (Utrecht — change place_query for full NL study area) …")
    POIAllocator.build_poi_cache("Utrecht, Netherlands", "nl_pois.gpkg")

    print()
    print("Done.")
