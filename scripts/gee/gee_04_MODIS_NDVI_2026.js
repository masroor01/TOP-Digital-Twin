// ============================================================
// GEE Script 04 — MODIS 16-Day NDVI/EVI 2026
// TOP Digital Twin: Tomato, Onion, Potato APMC Markets
// ============================================================
// Product: MOD13Q1 (Terra, 16-day, 250 m) + MYD13Q1 (Aqua, offset 8 days)
//          Merged to give ~8-day effective NDVI coverage.
//
// Output columns (match MODIS NDVI CSV format read by Script 14):
//   zone_id, crop, market, state, date, year, month, doy,
//   NDVI, EVI, n_valid_px
//
// NDVI scale: 0.0001 (stored as integer ×10000, divide to get 0–1)
// EVI  scale: 0.0001
// Quality:    SummaryQA == 0 (Good) or 1 (Marginal)  accepted;
//             SummaryQA == 2 (Snow/Ice) or 3 (Cloudy) rejected.
//
// Export: Google Drive → "TOP_Digital_Twin_GEE_2026" folder
//         One CSV per zone: {zone_id}_MODIS_NDVI_2026.csv
//
// NOTE: END is set to match the 2026-07-24 price/arrivals data you
// already have. If you're running this later with more 2026 weeks
// available, bump END forward before running. MODIS products also
// tend to lag real-time by a few weeks — check availability in the
// GEE catalog if the most recent composites are missing.
// ============================================================

var START = '2026-01-01';
var END   = '2026-07-25';  // exclusive — covers through 2026-07-24
var SCALE = 250;     // MOD13Q1 / MYD13Q1 native resolution 250 m
var FOLDER = 'TOP_Digital_Twin_GEE_2026';
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
// Quality mask: SummaryQA 0 = Good, 1 = Marginal (both usable)
// ──────────────────────────────────────────────────────────────
function maskQuality(img) {
  var qa = img.select('SummaryQA');
  var goodMask = qa.lte(1);  // 0 = Good, 1 = Marginal
  return img.updateMask(goodMask);
}

function scaleIndices(img) {
  return img
    .addBands(img.select('NDVI').multiply(0.0001).rename('NDVI_scaled'))
    .addBands(img.select('EVI').multiply(0.0001).rename('EVI_scaled'))
    .copyProperties(img, ['system:time_start']);
}

// ──────────────────────────────────────────────────────────────
// MOD13Q1 (Terra) + MYD13Q1 (Aqua) merged for ~8-day revisit
// ──────────────────────────────────────────────────────────────
var terra = ee.ImageCollection('MODIS/061/MOD13Q1')
  .filterDate(START, END)
  .select(['NDVI','EVI','SummaryQA'])
  .map(maskQuality)
  .map(scaleIndices);

var aqua = ee.ImageCollection('MODIS/061/MYD13Q1')
  .filterDate(START, END)
  .select(['NDVI','EVI','SummaryQA'])
  .map(maskQuality)
  .map(scaleIndices);

var modisNDVI = terra.merge(aqua).sort('system:time_start');

// ──────────────────────────────────────────────────────────────
// Export one CSV per zone
// ──────────────────────────────────────────────────────────────
zoneList.forEach(function(z) {
  var zone = zones.filter(ee.Filter.eq('zone_id', z.id)).first();
  var geom = zone.geometry();

  var composites = modisNDVI.map(function(img) {
    var stats = img.select(['NDVI_scaled','EVI_scaled']).reduceRegion({
      reducer  : ee.Reducer.mean(),
      geometry : geom,
      scale    : SCALE,
      maxPixels: 1e9
    });

    // Count valid (unmasked) pixels
    var nValid = img.select('NDVI_scaled').mask().reduceRegion({
      reducer  : ee.Reducer.sum(),
      geometry : geom,
      scale    : SCALE,
      maxPixels: 1e9
    });

    var dt = ee.Date(img.get('system:time_start'));
    return ee.Feature(null, {
      zone_id   : z.id,
      crop      : z.crop,
      market    : z.market,
      state     : z.state,
      date      : dt.format('YYYY-MM-dd'),
      year      : dt.get('year'),
      month     : dt.get('month'),
      doy       : dt.getRelative('day', 'year').add(1),
      NDVI      : stats.get('NDVI_scaled'),
      EVI       : stats.get('EVI_scaled'),
      n_valid_px: nValid.get('NDVI_scaled')
    });
  });

  // Keep only composites where NDVI was successfully computed
  composites = composites.filter(ee.Filter.notNull(['NDVI']));

  Export.table.toDrive({
    collection   : composites,
    description  : z.id + '_MODIS_NDVI_2026',
    folder       : FOLDER,
    fileNamePrefix: z.id + '_MODIS_NDVI_2026',
    fileFormat   : 'CSV',
    selectors    : ['zone_id','crop','market','state','date','year','month',
                    'doy','NDVI','EVI','n_valid_px']
  });
});

print('MODIS NDVI 2026 (Terra + Aqua): ' + zoneList.length + ' export tasks queued.');
print('Date range: ' + START + ' to ' + END);
print('Expected: ~fewer than 46 composites per zone for a Jan-Jul window (23 Terra + 23 Aqua for a full year).');
