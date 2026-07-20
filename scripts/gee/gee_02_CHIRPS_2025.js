// ============================================================
// GEE Script 02 — CHIRPS Pentad Rainfall 2025
// TOP Digital Twin: Tomato, Onion, Potato APMC Markets
// ============================================================
// Output columns (match CHIRPS CSV format read by Script 14):
//   zone_id, crop, market, state, date, year, month,
//   doy_pentad_start, rain_mean_mm, rain_sum_mm, rain_max_mm,
//   frac_excess_rain
//
// CHIRPS/PENTAD band "precipitation" = total mm for 5-day period.
//   rain_mean_mm   = spatial mean(precipitation) / 5  [mm/day]
//   rain_sum_mm    = spatial sum(precipitation) / 5   [pixel sum mm/day]
//   rain_max_mm    = spatial max(precipitation) / 5   [max daily mm in zone]
//   frac_excess_rain = fraction of zone pixels with precip > 50 mm/pentad
//                      (= > 10 mm/day threshold)
//
// Export: Google Drive → "TOP_Digital_Twin_GEE_2025" folder
//         One CSV per zone: {zone_id}_CHIRPS_2025.csv
// ============================================================

var START = '2025-01-01';
var END   = '2025-12-31';
var SCALE = 5566;    // CHIRPS native resolution 0.05° ≈ 5.5 km
var FOLDER = 'TOP_Digital_Twin_GEE_2025';
var BUFFER_M = 30000;
var EXCESS_THRESH_MM_PENTAD = 50;  // >50 mm/pentad = >10 mm/day = excess rain

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
// CHIRPS Pentad collection (5-day rainfall totals, mm)
// ──────────────────────────────────────────────────────────────
var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/PENTAD')
  .filterDate(START, END)
  .select('precipitation');  // mm per pentad (5-day total)

// Precompute derived bands
var chirpsDaily = chirps.map(function(img) {
  var precip = img.select('precipitation');      // mm/pentad
  var daily  = precip.divide(5);                // mm/day

  // Excess rain flag: pixel has > 10 mm/day = > 50 mm/pentad
  var excess = precip.gt(EXCESS_THRESH_MM_PENTAD).rename('excess');

  return daily.rename('daily_mm')
    .addBands(precip.rename('pentad_mm'))
    .addBands(excess)
    .set('system:time_start', img.get('system:time_start'))
    .set('date', img.date().format('YYYY-MM-dd'))
    .set('doy', img.date().getRelative('day', 'year').add(1));
});

// ──────────────────────────────────────────────────────────────
// Export one CSV per zone
// ──────────────────────────────────────────────────────────────
zoneList.forEach(function(z) {
  var zone = zones.filter(ee.Filter.eq('zone_id', z.id)).first();
  var geom = zone.geometry();

  var pentads = chirpsDaily.map(function(img) {
    var meanStats = img.select(['daily_mm','pentad_mm']).reduceRegion({
      reducer  : ee.Reducer.mean().combine(
                   ee.Reducer.sum(),  null, true).combine(
                   ee.Reducer.max(),  null, true),
      geometry : geom,
      scale    : SCALE,
      maxPixels: 1e6
    });
    var fracExcess = img.select('excess').reduceRegion({
      reducer  : ee.Reducer.mean(),
      geometry : geom,
      scale    : SCALE,
      maxPixels: 1e6
    });

    var dt = ee.Date(img.get('system:time_start'));
    return ee.Feature(null, {
      zone_id          : z.id,
      crop             : z.crop,
      market           : z.market,
      state            : z.state,
      date             : img.getString('date'),
      year             : dt.get('year'),
      month            : dt.get('month'),
      doy_pentad_start : img.get('doy'),
      // rain_mean_mm  = spatial mean daily rate (mm/day)
      rain_mean_mm     : meanStats.get('daily_mm_mean'),
      // rain_sum_mm   = pixel-sum of daily rates (= n_pixels × rain_mean_mm)
      rain_sum_mm      : meanStats.get('daily_mm_sum'),
      // rain_max_mm   = max daily rate in any zone pixel (mm/day)
      rain_max_mm      : meanStats.get('daily_mm_max'),
      // frac_excess_rain = fraction of pixels with > 10 mm/day
      frac_excess_rain : fracExcess.get('excess')
    });
  });

  Export.table.toDrive({
    collection   : pentads,
    description  : z.id + '_CHIRPS_2025',
    folder       : FOLDER,
    fileNamePrefix: z.id + '_CHIRPS_2025',
    fileFormat   : 'CSV',
    selectors    : ['zone_id','crop','market','state','date','year','month',
                    'doy_pentad_start','rain_mean_mm','rain_sum_mm',
                    'rain_max_mm','frac_excess_rain']
  });
});

print('CHIRPS 2025: ' + zoneList.length + ' export tasks queued.');
print('Date range: ' + START + ' to ' + END);
print('~73 pentads × 17 zones = ~1,241 rows total');
