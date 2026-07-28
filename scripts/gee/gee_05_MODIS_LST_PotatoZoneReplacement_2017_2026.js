// ============================================================
// GEE Script — MODIS 8-Day Land Surface Temperature, FULL HISTORY
// Potato Zone Replacement (see gee_01_ERA5_PotatoZoneReplacement_2017_2026.js
// for the full rationale). REVISED 2026-07-28: an earlier version of this
// script used a simplified day/night schema that didn't match Script 14's
// expected columns -- rewritten to match gee_05_MODIS_LST_2026.js exactly
// (LST_mean_C/LST_max_C/frac_above30-38/n_valid_px) so it loads through the
// same code path as every other zone.
//
// Product: MOD11A2 (Terra, 8-day daytime LST, 1 km)
//
// Output columns (match MODIS LST CSV format read by Script 14):
//   zone_id, crop, market, state, date, year, month, doy,
//   LST_mean_C, LST_max_C,
//   frac_above30, frac_above35, frac_above38,
//   n_valid_px
//
// Export: Google Drive -> "TOP_Digital_Twin_GEE_PotatoZoneReplacement" folder
//         One CSV per zone: {zone_id}_MODIS_LST_full_2017_2026.csv
// ============================================================

var START = '2017-01-01';
var END   = '2026-07-28';  // exclusive -- covers through 2026-07-27
var SCALE = 1000;    // MOD11A2 native resolution 1 km
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

function maskLSTquality(img) {
  var qc = img.select('QC_Day');
  var lsb = qc.bitwiseAnd(3);
  var goodMask = lsb.lt(2);
  return img.updateMask(goodMask);
}

function convertLST(img) {
  var lst_C = img.select('LST_Day_1km')
    .multiply(0.02)
    .subtract(273.15)
    .rename('LST_C');
  return img.addBands(lst_C)
    .copyProperties(img, ['system:time_start']);
}

var modisLST = ee.ImageCollection('MODIS/061/MOD11A2')
  .filterDate(START, END)
  .select(['LST_Day_1km', 'QC_Day'])
  .map(maskLSTquality)
  .map(convertLST);

zoneList.forEach(function(z) {
  var zone = zones.filter(ee.Filter.eq('zone_id', z.id)).first();
  var geom = zone.geometry();

  var composites = modisLST.map(function(img) {
    var lst = img.select('LST_C');

    var meanMax = lst.reduceRegion({
      reducer  : ee.Reducer.mean().combine(ee.Reducer.max(), null, true),
      geometry : geom,
      scale    : SCALE,
      maxPixels: 1e9
    });

    var f30 = lst.gt(30).reduceRegion({
      reducer: ee.Reducer.mean(), geometry: geom, scale: SCALE, maxPixels: 1e9
    });
    var f35 = lst.gt(35).reduceRegion({
      reducer: ee.Reducer.mean(), geometry: geom, scale: SCALE, maxPixels: 1e9
    });
    var f38 = lst.gt(38).reduceRegion({
      reducer: ee.Reducer.mean(), geometry: geom, scale: SCALE, maxPixels: 1e9
    });

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

  composites = composites.filter(ee.Filter.notNull(['LST_mean_C']));

  Export.table.toDrive({
    collection   : composites,
    description  : z.id + '_MODIS_LST_full_2017_2026',
    folder       : FOLDER,
    fileNamePrefix: z.id + '_MODIS_LST_full_2017_2026',
    fileFormat   : 'CSV',
    selectors    : ['zone_id','crop','market','state','date','year','month','doy',
                    'LST_mean_C','LST_max_C',
                    'frac_above30','frac_above35','frac_above38','n_valid_px']
  });
});

print('MODIS LST potato-zone replacement (Terra MOD11A2), full history: ' + zoneList.length + ' export tasks queued.');
print('Date range: ' + START + ' to ' + END);
