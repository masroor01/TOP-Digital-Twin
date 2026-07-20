// ============================================================
// GEE Script 05 — MODIS 8-Day Land Surface Temperature 2025
// TOP Digital Twin: Tomato, Onion, Potato APMC Markets
// ============================================================
// Product: MOD11A2 (Terra, 8-day daytime LST, 1 km)
//
// Output columns (match MODIS LST CSV format read by Script 14):
//   zone_id, crop, market, state, date, year, month, doy,
//   LST_mean_C, LST_max_C,
//   frac_above30, frac_above35, frac_above38,
//   n_valid_px
//
// Conversion: LST_Day_1km [DN × 0.02 K] → Celsius = DN×0.02 − 273.15
// Quality:    QC_Day bits 0-1 = 00 → "Produced, good quality"
//             Reject bits 0-1 = 01,10,11 (produced but uncertain / not produced)
//
// Export: Google Drive → "TOP_Digital_Twin_GEE_2025" folder
//         One CSV per zone: {zone_id}_MODIS_LST_2025.csv
// ============================================================

var START = '2025-01-01';
var END   = '2025-12-31';
var SCALE = 1000;    // MOD11A2 native resolution 1 km
var FOLDER = 'TOP_Digital_Twin_GEE_2025';
var BUFFER_M = 30000;

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
// Quality mask: QC_Day bits 0-1
//   00 = Produced, good quality → keep
//   01 = Produced, other quality → keep (marginal but usable)
//   10 = Not produced due to cloud → reject
//   11 = Not produced due to other → reject
// Bit mask: bits 0-1 < 2 means "produced" (good or marginal)
// ──────────────────────────────────────────────────────────────
function maskLSTquality(img) {
  var qc = img.select('QC_Day');
  // Extract bits 0-1: if bits 0-1 < 2, LST was produced
  var lsb = qc.bitwiseAnd(3);  // keep only lowest 2 bits (bits 0-1)
  var goodMask = lsb.lt(2);    // 0=good, 1=marginal → accept
  return img.updateMask(goodMask);
}

// ──────────────────────────────────────────────────────────────
// Convert DN to Celsius: scale=0.02 K, subtract 273.15
// ──────────────────────────────────────────────────────────────
function convertLST(img) {
  var lst_C = img.select('LST_Day_1km')
    .multiply(0.02)
    .subtract(273.15)
    .rename('LST_C');
  return img.addBands(lst_C)
    .copyProperties(img, ['system:time_start']);
}

// ──────────────────────────────────────────────────────────────
// MOD11A2: Terra 8-day LST composite
// ──────────────────────────────────────────────────────────────
var modisLST = ee.ImageCollection('MODIS/061/MOD11A2')
  .filterDate(START, END)
  .select(['LST_Day_1km', 'QC_Day'])
  .map(maskLSTquality)
  .map(convertLST);

// ──────────────────────────────────────────────────────────────
// Export one CSV per zone
// ──────────────────────────────────────────────────────────────
zoneList.forEach(function(z) {
  var zone = zones.filter(ee.Filter.eq('zone_id', z.id)).first();
  var geom = zone.geometry();

  var composites = modisLST.map(function(img) {
    var lst = img.select('LST_C');

    // Mean and max LST in zone
    var meanMax = lst.reduceRegion({
      reducer  : ee.Reducer.mean().combine(ee.Reducer.max(), null, true),
      geometry : geom,
      scale    : SCALE,
      maxPixels: 1e9
    });

    // Fraction of zone pixels exceeding each threshold
    var f30 = lst.gt(30).reduceRegion({
      reducer: ee.Reducer.mean(), geometry: geom, scale: SCALE, maxPixels: 1e9
    });
    var f35 = lst.gt(35).reduceRegion({
      reducer: ee.Reducer.mean(), geometry: geom, scale: SCALE, maxPixels: 1e9
    });
    var f38 = lst.gt(38).reduceRegion({
      reducer: ee.Reducer.mean(), geometry: geom, scale: SCALE, maxPixels: 1e9
    });

    // Valid pixel count (only unmasked pixels counted)
    var nValid = lst.mask().reduceRegion({
      reducer: ee.Reducer.sum(), geometry: geom, scale: SCALE, maxPixels: 1e9
    });

    var dt = ee.Date(img.get('system:time_start'));
    return ee.Feature(null, {
      zone_id      : z.id,
      crop         : z.crop,
      market       : z.market,
      state        : z.state,
      date         : dt.format('YYYY-MM-dd'),
      year         : dt.get('year'),
      month        : dt.get('month'),
      doy          : dt.getRelative('day', 'year').add(1),
      LST_mean_C   : meanMax.get('LST_C_mean'),
      LST_max_C    : meanMax.get('LST_C_max'),
      frac_above30 : f30.get('LST_C'),
      frac_above35 : f35.get('LST_C'),
      frac_above38 : f38.get('LST_C'),
      n_valid_px   : nValid.get('LST_C')
    });
  });

  // Remove composites fully obscured by cloud (no valid LST)
  composites = composites.filter(ee.Filter.notNull(['LST_mean_C']));

  Export.table.toDrive({
    collection   : composites,
    description  : z.id + '_MODIS_LST_2025',
    folder       : FOLDER,
    fileNamePrefix: z.id + '_MODIS_LST_2025',
    fileFormat   : 'CSV',
    selectors    : ['zone_id','crop','market','state','date','year','month','doy',
                    'LST_mean_C','LST_max_C',
                    'frac_above30','frac_above35','frac_above38','n_valid_px']
  });
});

print('MODIS LST 2025 (Terra MOD11A2): ' + zoneList.length + ' export tasks queued.');
print('Expected: ~46 composites per zone (8-day, cloud-filtered)');
