// ============================================================
// GEE Script — MODIS 16-Day NDVI/EVI, FULL HISTORY
// Potato Zone Replacement (see gee_01_ERA5_PotatoZoneReplacement_2017_2026.js
// for the full rationale). REVISED 2026-07-28: an earlier version of this
// script used a simplified schema (MOD13Q1 only, no n_valid_px) that didn't
// match Script 14's expected columns -- rewritten to match
// gee_04_MODIS_NDVI_2026.js exactly (Terra+Aqua merge, n_valid_px) so it
// loads through the same code path as every other zone.
//
// Product: MOD13Q1 (Terra, 16-day, 250 m) + MYD13Q1 (Aqua, offset 8 days)
//          Merged to give ~8-day effective NDVI coverage.
//
// Output columns (match MODIS NDVI CSV format read by Script 14):
//   zone_id, crop, market, state, date, year, month, doy,
//   NDVI, EVI, n_valid_px
//
// Export: Google Drive -> "TOP_Digital_Twin_GEE_PotatoZoneReplacement" folder
//         One CSV per zone: {zone_id}_MODIS_NDVI_full_2017_2026.csv
// ============================================================

var START = '2017-01-01';
var END   = '2026-07-28';  // exclusive -- covers through 2026-07-27
var SCALE = 250;     // MOD13Q1 / MYD13Q1 native resolution 250 m
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

  composites = composites.filter(ee.Filter.notNull(['NDVI']));

  Export.table.toDrive({
    collection   : composites,
    description  : z.id + '_MODIS_NDVI_full_2017_2026',
    folder       : FOLDER,
    fileNamePrefix: z.id + '_MODIS_NDVI_full_2017_2026',
    fileFormat   : 'CSV',
    selectors    : ['zone_id','crop','market','state','date','year','month',
                    'doy','NDVI','EVI','n_valid_px']
  });
});

print('MODIS NDVI potato-zone replacement (Terra+Aqua), full history: ' + zoneList.length + ' export tasks queued.');
print('Date range: ' + START + ' to ' + END);
