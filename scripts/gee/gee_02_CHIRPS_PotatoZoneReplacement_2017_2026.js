// ============================================================
// GEE Script — CHIRPS Pentad Rainfall, FULL HISTORY
// Potato Zone Replacement (see gee_01_ERA5_PotatoZoneReplacement_2017_2026.js
// for the full rationale): P1/P2/P3 relocated to Darjeeling / Diamond
// Harbour / Dehradun -- real, currently-modeled potato markets --
// replacing the unreachable Agra/Farrukhabad/Jalandhar zones.
//
// Output columns (match CHIRPS CSV format read by Script 14):
//   zone_id, crop, market, state, date, year, month,
//   doy_pentad_start, rain_mean_mm, rain_sum_mm, rain_max_mm,
//   frac_excess_rain
//
// Export: Google Drive -> "TOP_Digital_Twin_GEE_PotatoZoneReplacement" folder
//         One CSV per zone: {zone_id}_CHIRPS_full_2017_2026.csv
// ============================================================

var START = '2017-01-01';
var END   = '2026-07-28';  // exclusive -- covers through 2026-07-27
var SCALE = 5566;    // CHIRPS native resolution 0.05 deg ~= 5.5 km
var FOLDER = 'TOP_Digital_Twin_GEE_PotatoZoneReplacement';
var BUFFER_M = 30000;
var EXCESS_THRESH_MM_PENTAD = 50;  // >50 mm/pentad = >10 mm/day = excess rain

var zoneList = [
  {id:'P1_Darjeeling',     crop:'Potato', market:'Darjeeling APMC',                      state:'West Bengal',  lon:88.263176,  lat:27.037755},
  {id:'P2_DiamondHarbour', crop:'Potato', market:'Diamond Harbour(South 24-pgs) APMC',   state:'West Bengal',  lon:88.189488,  lat:22.192689},
  {id:'P3_Dehradun',       crop:'Potato', market:'Dehradoon APMC',                       state:'Uttarakhand',  lon:78.043681,  lat:30.325565}
];

var zones = ee.FeatureCollection(zoneList.map(function(z) {
  return ee.Feature(
    ee.Geometry.Point([z.lon, z.lat]).buffer(BUFFER_M),
    {zone_id: z.id, crop: z.crop, market: z.market, state: z.state}
  );
}));

var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/PENTAD')
  .filterDate(START, END)
  .select('precipitation');

var chirpsDaily = chirps.map(function(img) {
  var precip = img.select('precipitation');
  var daily  = precip.divide(5);

  var excess = precip.gt(EXCESS_THRESH_MM_PENTAD).rename('excess');

  return daily.rename('daily_mm')
    .addBands(precip.rename('pentad_mm'))
    .addBands(excess)
    .set('system:time_start', img.get('system:time_start'))
    .set('date', img.date().format('YYYY-MM-dd'))
    .set('doy', img.date().getRelative('day', 'year').add(1));
});

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
      rain_mean_mm     : meanStats.get('daily_mm_mean'),
      rain_sum_mm      : meanStats.get('daily_mm_sum'),
      rain_max_mm      : meanStats.get('daily_mm_max'),
      frac_excess_rain : fracExcess.get('excess')
    });
  });

  Export.table.toDrive({
    collection   : pentads,
    description  : z.id + '_CHIRPS_full_2017_2026',
    folder       : FOLDER,
    fileNamePrefix: z.id + '_CHIRPS_full_2017_2026',
    fileFormat   : 'CSV',
    selectors    : ['zone_id','crop','market','state','date','year','month',
                    'doy_pentad_start','rain_mean_mm','rain_sum_mm',
                    'rain_max_mm','frac_excess_rain']
  });
});

print('CHIRPS potato-zone replacement, full history: ' + zoneList.length + ' export tasks queued.');
print('Date range: ' + START + ' to ' + END);
