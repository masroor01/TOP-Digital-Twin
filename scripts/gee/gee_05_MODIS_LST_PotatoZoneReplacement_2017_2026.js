// ============================================================
// GEE Script — MODIS Land Surface Temperature, 8-day composites, FULL HISTORY
// Potato Zone Replacement (see gee_01_ERA5_PotatoZoneReplacement_2017_2026.js
// for the full rationale). MOD11A2 is a native 8-day composite product,
// so no manual windowing is needed -- filter to the full date range and
// reduce each native composite over the zone geometry.
//
// Output columns (match MODIS LST CSV format read by Script 14):
//   zone_id, crop, market, state, date, year, doy,
//   LST_day_C, LST_night_C
//
// Export: Google Drive -> "TOP_Digital_Twin_GEE_PotatoZoneReplacement" folder
//         One CSV per zone: {zone_id}_MODIS_LST_full_2017_2026.csv
// ============================================================

var START = '2017-01-01';
var END   = '2026-07-28';  // exclusive -- covers through 2026-07-27
var SCALE = 1000;   // MOD11A2 native resolution
var FOLDER = 'TOP_Digital_Twin_GEE_PotatoZoneReplacement';
var BUFFER_M = 30000;

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

var modis = ee.ImageCollection('MODIS/061/MOD11A2')
  .filterDate(START, END)
  .select(['LST_Day_1km', 'LST_Night_1km']);

var modisC = modis.map(function(img) {
  var day   = img.select('LST_Day_1km').multiply(0.02).subtract(273.15).rename('LST_day_C');
  var night = img.select('LST_Night_1km').multiply(0.02).subtract(273.15).rename('LST_night_C');
  return day.addBands(night)
    .set('system:time_start', img.get('system:time_start'))
    .set('date', img.date().format('YYYY-MM-dd'));
});

zoneList.forEach(function(z) {
  var zone = zones.filter(ee.Filter.eq('zone_id', z.id)).first();
  var geom = zone.geometry();

  var composites = modisC.map(function(img) {
    var stats = img.reduceRegion({
      reducer  : ee.Reducer.mean(),
      geometry : geom,
      scale    : SCALE,
      maxPixels: 1e9
    });

    var dt = ee.Date(img.get('system:time_start'));
    return ee.Feature(null, {
      zone_id     : z.id,
      crop        : z.crop,
      market      : z.market,
      state       : z.state,
      date        : img.getString('date'),
      year        : dt.get('year'),
      doy         : dt.getRelative('day', 'year').add(1),
      LST_day_C   : stats.get('LST_day_C'),
      LST_night_C : stats.get('LST_night_C')
    });
  });

  Export.table.toDrive({
    collection   : composites,
    description  : z.id + '_MODIS_LST_full_2017_2026',
    folder       : FOLDER,
    fileNamePrefix: z.id + '_MODIS_LST_full_2017_2026',
    fileFormat   : 'CSV',
    selectors    : ['zone_id','crop','market','state','date','year','doy','LST_day_C','LST_night_C']
  });
});

print('MODIS LST potato-zone replacement, full history: ' + zoneList.length + ' export tasks queued.');
print('Date range: ' + START + ' to ' + END + ' (~430 eight-day composites/zone over ~9.5 years).');
