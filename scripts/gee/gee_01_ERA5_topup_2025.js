// ============================================================
// GEE Script 01 — ERA5-Land Daily Temperature Topup 2025
// TOP Digital Twin: Tomato, Onion, Potato APMC Markets
// ============================================================
// Output columns (match ERA5topup CSV format read by Script 14):
//   zone_id, crop, market, state, date, year, month, doy,
//   Tmax_C, Tmin_C, Tmean_C,
//   flag_above30, flag_above35, flag_above38
//
// Export: Google Drive → "TOP_Digital_Twin_GEE_2025" folder
//         One CSV per zone: {zone_id}_ERA5topup_2025.csv
//
// Run:  Paste into code.earthengine.google.com → Run
//       Click "Tasks" tab → Run all 17 export tasks
// ============================================================

var START = '2025-01-01';
var END   = '2025-12-31';
var SCALE = 11132;   // ERA5-Land native resolution ~0.1° ≈ 11 km
var FOLDER = 'TOP_Digital_Twin_GEE_2025';
var BUFFER_M = 30000;  // 30 km production zone radius

// ──────────────────────────────────────────────────────────────
// Zone definitions: centroid coords + metadata
// Each zone covers the surrounding agricultural production area.
// Centroid = APMC market location; buffer = 30 km hinterland.
// ──────────────────────────────────────────────────────────────
var zoneList = [
  // Tomato zones
  {id:'T1_Kolar',        crop:'Tomato', market:'Kolar APMC',        state:'Karnataka',       lon:78.1320, lat:13.1390},
  {id:'T2_Madanapalle',  crop:'Tomato', market:'Madanapalle APMC',  state:'Andhra Pradesh',  lon:78.5025, lat:13.5504},
  {id:'T3_Nashik_Tomato',crop:'Tomato', market:'Nashik APMC',       state:'Maharashtra',     lon:73.7898, lat:19.9975},
  {id:'T4_Solan',        crop:'Tomato', market:'Solan APMC',        state:'Himachal Pradesh',lon:77.1167, lat:30.9045},
  {id:'T5_Navsari',      crop:'Tomato', market:'Navsari APMC',      state:'Gujarat',         lon:72.9031, lat:20.9476},
  // Onion zones
  {id:'O1_Lasalgaon',    crop:'Onion',  market:'Lasalgaon APMC',    state:'Maharashtra',     lon:74.0088, lat:20.4061},
  {id:'O2_Pimpalgaon',   crop:'Onion',  market:'Pimpalgaon APMC',   state:'Maharashtra',     lon:74.0167, lat:20.0833},
  {id:'O3_Mahuva',       crop:'Onion',  market:'Mahuva APMC',       state:'Gujarat',         lon:71.7744, lat:21.0888},
  {id:'O6_Hubli',        crop:'Onion',  market:'Hubli APMC',        state:'Karnataka',       lon:75.1239, lat:15.3647},
  {id:'O7_Solapur',      crop:'Onion',  market:'Solapur APMC',      state:'Maharashtra',     lon:75.9064, lat:17.6854},
  {id:'O8_Manmad',       crop:'Onion',  market:'Manmad APMC',       state:'Maharashtra',     lon:74.4367, lat:20.2500},
  {id:'O9_Kurnool',      crop:'Onion',  market:'Kurnool APMC',      state:'Andhra Pradesh',  lon:78.0373, lat:15.8281},
  {id:'O10_Gondal',      crop:'Onion',  market:'Gondal APMC',       state:'Gujarat',         lon:70.7980, lat:21.9608},
  // Potato zones
  {id:'P1_Agra',         crop:'Potato', market:'Agra APMC',         state:'Uttar Pradesh',   lon:78.0081, lat:27.1767},
  {id:'P2_Farrukhabad',  crop:'Potato', market:'Farrukhabad APMC',  state:'Uttar Pradesh',   lon:79.5800, lat:27.3900},
  {id:'P3_Jalandhar',    crop:'Potato', market:'Jalandhar APMC',    state:'Punjab',          lon:75.5762, lat:31.3260},
  {id:'P4_Bardhaman',    crop:'Potato', market:'Bardhaman APMC',    state:'West Bengal',     lon:87.8550, lat:23.2330}
];

// Build FeatureCollection with 30 km buffer geometries
var zones = ee.FeatureCollection(zoneList.map(function(z) {
  return ee.Feature(
    ee.Geometry.Point([z.lon, z.lat]).buffer(BUFFER_M),
    {zone_id: z.id, crop: z.crop, market: z.market, state: z.state}
  );
}));

// ──────────────────────────────────────────────────────────────
// ERA5-Land daily aggregated collection
// Bands used:
//   temperature_2m_max  (K) → Tmax_C = val - 273.15
//   temperature_2m_min  (K) → Tmin_C = val - 273.15
//   temperature_2m      (K) → Tmean_C = val - 273.15
// ──────────────────────────────────────────────────────────────
var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
  .filterDate(START, END)
  .select(['temperature_2m_max', 'temperature_2m_min', 'temperature_2m']);

// Convert each image to Celsius and compute threshold flags
var era5_C = era5.map(function(img) {
  var tmax  = img.select('temperature_2m_max').subtract(273.15).rename('Tmax_C');
  var tmin  = img.select('temperature_2m_min').subtract(273.15).rename('Tmin_C');
  var tmean = img.select('temperature_2m').subtract(273.15).rename('Tmean_C');

  // Spatial fraction of zone pixels exceeding threshold
  // (ERA5 at 11 km: for a 30 km buffer, ~6–9 pixels; partial edge pixels give non-integer fractions)
  var f30 = tmax.gt(30).rename('flag_above30');
  var f35 = tmax.gt(35).rename('flag_above35');
  var f38 = tmax.gt(38).rename('flag_above38');

  return tmax.addBands([tmin, tmean, f30, f35, f38])
    .set('system:time_start', img.get('system:time_start'))
    .set('date', img.date().format('YYYY-MM-dd'));
});

// ──────────────────────────────────────────────────────────────
// Extract one row per zone per day, export per-zone CSV
// ──────────────────────────────────────────────────────────────
zoneList.forEach(function(z) {
  var zone = zones.filter(ee.Filter.eq('zone_id', z.id)).first();
  var geom = zone.geometry();

  var daily = era5_C.map(function(img) {
    var stats = img.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: geom,
      scale: SCALE,
      maxPixels: 1e6
    });

    var dt = ee.Date(img.get('system:time_start'));
    return ee.Feature(null, {
      zone_id        : z.id,
      crop           : z.crop,
      market         : z.market,
      state          : z.state,
      date           : img.getString('date'),
      year           : dt.get('year'),
      month          : dt.get('month'),
      doy            : dt.getRelative('day', 'year').add(1),
      Tmax_C         : stats.get('Tmax_C'),
      Tmin_C         : stats.get('Tmin_C'),
      Tmean_C        : stats.get('Tmean_C'),
      flag_above30   : stats.get('flag_above30'),
      flag_above35   : stats.get('flag_above35'),
      flag_above38   : stats.get('flag_above38')
    });
  });

  Export.table.toDrive({
    collection  : daily,
    description : z.id + '_ERA5topup_2025',
    folder      : FOLDER,
    fileNamePrefix: z.id + '_ERA5topup_2025',
    fileFormat  : 'CSV',
    selectors   : ['zone_id','crop','market','state','date','year','month','doy',
                   'Tmax_C','Tmin_C','Tmean_C','flag_above30','flag_above35','flag_above38']
  });
});

print('ERA5 topup 2025: ' + zoneList.length + ' export tasks queued.');
print('Date range: ' + START + ' to ' + END);
print('Open the Tasks tab and click Run on each task.');
