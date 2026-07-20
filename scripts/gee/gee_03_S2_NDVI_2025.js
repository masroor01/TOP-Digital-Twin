// ============================================================
// GEE Script 03 — Sentinel-2 NDVI/EVI Bimonthly Composites 2025
// TOP Digital Twin: Tomato, Onion, Potato APMC Markets
// ============================================================
// Output columns (match S2 CSV format read by Script 14):
//   zone_id, crop, market, state, date_start, date_end, year,
//   doy_start, n_scenes, NDVI, EVI, valid_px_frac
//
// Compositing: 15-day windows (1–15 and 16–end of each month)
//   → ~24 composites per zone per year
// Cloud masking: SCL band (S2_SR_HARMONIZED)
//   Keep SCL values: 4 (veg), 5 (bare soil), 6 (water),
//                    7 (unclassified), 11 (snow/ice)
//   Mask: 1 (saturated), 2 (dark), 3 (cloud shadow),
//          8 (med cloud), 9 (high cloud), 10 (thin cirrus)
//
// valid_px_frac = fraction of zone pixels that are cloud-free
//                 (clear scenes only; NaN composites are skipped)
//
// Export: Google Drive → "TOP_Digital_Twin_GEE_2025" folder
//         One CSV per zone: {zone_id}_S2_NDVI_2025.csv
// ============================================================

var YEAR  = 2025;
var SCALE = 10;      // S2 native resolution 10 m (NDVI bands B4, B8)
var FOLDER = 'TOP_Digital_Twin_GEE_2025';
var BUFFER_M = 30000;
var MIN_VALID_FRAC = 0.05;  // skip composite if < 5% valid pixels

var zoneList = [
  {id:'T1_Kolar',        crop:'Tomato', market:'Kolar APMC',        state:'Karnataka',       lon:78.1320, lat:13.1390},
  {id:'T2_Madanapalle',  crop:'Tomato', market:'Madanapalle APMC',  state:'Andhra Pradesh',  lon:78.5025, lat:13.5504},
  {id:'T3_Nashik_Tomato',crop:'Tomato', market:'Nashik APMC',       state:'Maharashtra',     lon:73.7898, lat:19.9975},
  {id:'T4_Solan',        crop:'Tomato', market:'Solan APMC',        state:'Himachal Pradesh',lon:77.1167, lat:30.9045},
  {id:'T5_Navsari',      crop:'Tomato', market:'Navsari APMC',      state:'Gujarat',         lon:72.9031, lat:20.9476},
  {id:'O1_Lasalgaon',    crop:'Onion',  market:'Lasalgaon APMC',    state:'Maharashtra',     lon:74.0088, lat:20.4061},
  {id:'O2_Pimpalgaon',   crop:'Onion',  market:'Pimpalgaon APMC',   state:'Maharashtra',     lon:74.0167, lat:20.0833},
  {id:'O3_Mahuva',       crop:'Onion',  market:'Mahuva APMC',       state:'Gujarat',         lon:71.7744, lat:21.0888},
  {id:'O6_Hubli',        crop:'Onion',  market:'Hubli APMC',        state:'Karnataka',       lon:75.1239, lat:15.3647},
  {id:'O7_Solapur',      crop:'Onion',  market:'Solapur APMC',      state:'Maharashtra',     lon:75.9064, lat:17.6854},
  {id:'O8_Manmad',       crop:'Onion',  market:'Manmad APMC',       state:'Maharashtra',     lon:74.4367, lat:20.2500},
  {id:'O9_Kurnool',      crop:'Onion',  market:'Kurnool APMC',      state:'Andhra Pradesh',  lon:78.0373, lat:15.8281},
  {id:'O10_Gondal',      crop:'Onion',  market:'Gondal APMC',       state:'Gujarat',         lon:70.7980, lat:21.9608},
  {id:'P1_Agra',         crop:'Potato', market:'Agra APMC',         state:'Uttar Pradesh',   lon:78.0081, lat:27.1767},
  {id:'P2_Farrukhabad',  crop:'Potato', market:'Farrukhabad APMC',  state:'Uttar Pradesh',   lon:79.5800, lat:27.3900},
  {id:'P3_Jalandhar',    crop:'Potato', market:'Jalandhar APMC',    state:'Punjab',          lon:75.5762, lat:31.3260},
  {id:'P4_Bardhaman',    crop:'Potato', market:'Bardhaman APMC',    state:'West Bengal',     lon:87.8550, lat:23.2330}
];

var zones = ee.FeatureCollection(zoneList.map(function(z) {
  return ee.Feature(
    ee.Geometry.Point([z.lon, z.lat]).buffer(BUFFER_M),
    {zone_id: z.id, crop: z.crop, market: z.market, state: z.state}
  );
}));

// ──────────────────────────────────────────────────────────────
// Cloud masking using Scene Classification Layer (SCL)
// Bit flags for clear pixels: SCL 4,5,6,7,11
// ──────────────────────────────────────────────────────────────
function maskS2clouds(img) {
  var scl = img.select('SCL');
  var clearMask = scl.eq(4)   // Vegetation
    .or(scl.eq(5))            // Bare soil / not vegetated
    .or(scl.eq(6))            // Water
    .or(scl.eq(7))            // Unclassified
    .or(scl.eq(11));          // Snow / Ice
  return img.updateMask(clearMask).set('clear_mask', clearMask);
}

// ──────────────────────────────────────────────────────────────
// Compute NDVI and EVI from S2 SR bands (already scaled 0–1 in
// HARMONIZED collection; B2=Blue, B4=Red, B8=NIR)
// ──────────────────────────────────────────────────────────────
function addIndices(img) {
  var b4  = img.select('B4');   // Red
  var b8  = img.select('B8');   // NIR
  var b2  = img.select('B2');   // Blue

  var ndvi = b8.subtract(b4).divide(b8.add(b4)).rename('NDVI');
  var evi  = b8.subtract(b4)
    .divide(b8.add(b4.multiply(6)).subtract(b2.multiply(7.5)).add(1))
    .multiply(2.5)
    .rename('EVI');

  return img.addBands([ndvi, evi]);
}

// ──────────────────────────────────────────────────────────────
// Build 15-day composite windows for 2025
// Windows: 1-15 and 16-end for each month
// ──────────────────────────────────────────────────────────────
var windows = [];
var monthDays = [31,28,31,30,31,30,31,31,30,31,30,31]; // 2025 is not leap year
for (var m = 1; m <= 12; m++) {
  var dEnd = monthDays[m-1];
  // First half: 1–15
  windows.push({
    start: ee.Date.fromYMD(YEAR, m, 1),
    end:   ee.Date.fromYMD(YEAR, m, 15).advance(1, 'day'),
    dateStart: YEAR + '-' + (m<10?'0':'') + m + '-01',
    dateEnd:   YEAR + '-' + (m<10?'0':'') + m + '-15'
  });
  // Second half: 16–end
  windows.push({
    start: ee.Date.fromYMD(YEAR, m, 16),
    end:   ee.Date.fromYMD(YEAR, m, dEnd).advance(1, 'day'),
    dateStart: YEAR + '-' + (m<10?'0':'') + m + '-16',
    dateEnd:   YEAR + '-' + (m<10?'0':'') + m + '-' + dEnd
  });
}

// Load S2 SR Harmonized collection for 2025
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterDate('2025-01-01', '2025-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 95))
  .map(maskS2clouds)
  .map(addIndices);

// ──────────────────────────────────────────────────────────────
// Export one CSV per zone
// ──────────────────────────────────────────────────────────────
zoneList.forEach(function(z) {
  var zone = zones.filter(ee.Filter.eq('zone_id', z.id)).first();
  var geom = zone.geometry();

  var composites = ee.FeatureCollection(windows.map(function(w) {
    var sub = s2.filterDate(w.start, w.end).filterBounds(geom);
    var nScenes = sub.size();

    // Guard: if collection is empty (all scenes cloudy), substitute a fully-masked
    // placeholder image that carries NDVI and EVI bands. reduceRegion then returns
    // null for both, and the notNull filter below drops the row cleanly.
    // Without this, sub.median() on an empty collection returns a zero-band image,
    // which causes "Band pattern 'NDVI' was applied to an image with no bands".
    var placeholder = ee.Image.constant([0, 0]).rename(['NDVI', 'EVI']).selfMask();
    var comp = ee.Image(ee.Algorithms.If(nScenes.gt(0), sub.median(), placeholder));

    // Spatial mean NDVI and EVI over zone (masked pixels excluded)
    var meanStats = comp.select(['NDVI','EVI']).reduceRegion({
      reducer  : ee.Reducer.mean(),
      geometry : geom,
      scale    : SCALE,
      maxPixels: 1e13
    });

    // valid_px_frac: fraction of zone pixels with valid (unmasked) data
    var validFrac = comp.select('NDVI').mask().reduceRegion({
      reducer  : ee.Reducer.mean(),
      geometry : geom,
      scale    : SCALE,
      maxPixels: 1e13
    });

    var doyStart = ee.Date(w.start).getRelative('day', 'year').add(1);

    return ee.Feature(null, {
      zone_id      : z.id,
      crop         : z.crop,
      market       : z.market,
      state        : z.state,
      date_start   : w.dateStart,
      date_end     : w.dateEnd,
      year         : YEAR,
      doy_start    : doyStart,
      n_scenes     : nScenes,
      NDVI         : meanStats.get('NDVI'),
      EVI          : meanStats.get('EVI'),
      valid_px_frac: validFrac.get('NDVI')
    });
  }));

  // Filter out composites with no valid pixels
  composites = composites.filter(ee.Filter.notNull(['NDVI']));

  Export.table.toDrive({
    collection   : composites,
    description  : z.id + '_S2_NDVI_2025',
    folder       : FOLDER,
    fileNamePrefix: z.id + '_S2_NDVI_2025',
    fileFormat   : 'CSV',
    selectors    : ['zone_id','crop','market','state','date_start','date_end',
                    'year','doy_start','n_scenes','NDVI','EVI','valid_px_frac']
  });
});

print('S2 NDVI 2025: ' + zoneList.length + ' export tasks queued.');
print('Up to 24 composites per zone (15-day windows, cloud-filtered).');
