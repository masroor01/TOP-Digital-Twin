// ============================================================
// GEE Script — MODIS NDVI/EVI 16-day Composites, FULL HISTORY
// Potato Zone Replacement (see gee_01_ERA5_PotatoZoneReplacement_2017_2026.js
// for the full rationale). MOD13Q1 is a native 16-day composite product,
// so — unlike gee_03 (S2) — no manual windowing/compositing is needed;
// we simply filter the collection to the full date range and reduce
// each native composite image over the zone geometry.
//
// Output columns (match MODIS NDVI CSV format read by Script 14):
//   zone_id, crop, market, state, date, year, doy, NDVI, EVI
//
// Export: Google Drive -> "TOP_Digital_Twin_GEE_PotatoZoneReplacement" folder
//         One CSV per zone: {zone_id}_MODIS_NDVI_full_2017_2026.csv
// ============================================================

var START = '2017-01-01';
var END   = '2026-07-28';  // exclusive -- covers through 2026-07-27
var SCALE = 250;    // MOD13Q1 native resolution
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

var modis = ee.ImageCollection('MODIS/061/MOD13Q1')
  .filterDate(START, END)
  .select(['NDVI', 'EVI']);

var modisScaled = modis.map(function(img) {
  var ndvi = img.select('NDVI').multiply(0.0001).rename('NDVI');
  var evi  = img.select('EVI').multiply(0.0001).rename('EVI');
  return ndvi.addBands(evi)
    .set('system:time_start', img.get('system:time_start'))
    .set('date', img.date().format('YYYY-MM-dd'));
});

zoneList.forEach(function(z) {
  var zone = zones.filter(ee.Filter.eq('zone_id', z.id)).first();
  var geom = zone.geometry();

  var composites = modisScaled.map(function(img) {
    var stats = img.reduceRegion({
      reducer  : ee.Reducer.mean(),
      geometry : geom,
      scale    : SCALE,
      maxPixels: 1e9
    });

    var dt = ee.Date(img.get('system:time_start'));
    return ee.Feature(null, {
      zone_id : z.id,
      crop    : z.crop,
      market  : z.market,
      state   : z.state,
      date    : img.getString('date'),
      year    : dt.get('year'),
      doy     : dt.getRelative('day', 'year').add(1),
      NDVI    : stats.get('NDVI'),
      EVI     : stats.get('EVI')
    });
  });

  Export.table.toDrive({
    collection   : composites,
    description  : z.id + '_MODIS_NDVI_full_2017_2026',
    folder       : FOLDER,
    fileNamePrefix: z.id + '_MODIS_NDVI_full_2017_2026',
    fileFormat   : 'CSV',
    selectors    : ['zone_id','crop','market','state','date','year','doy','NDVI','EVI']
  });
});

print('MODIS NDVI potato-zone replacement, full history: ' + zoneList.length + ' export tasks queued.');
print('Date range: ' + START + ' to ' + END + ' (~230 sixteen-day composites/zone over ~9.5 years).');
