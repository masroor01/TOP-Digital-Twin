// ============================================================
// GEE Script — Sentinel-2 NDVI/EVI Bimonthly Composites, FULL HISTORY
// Potato Zone Replacement (see gee_01_ERA5_PotatoZoneReplacement_2017_2026.js
// for the full rationale). Unlike the single-year topup scripts, this
// loops over MULTIPLE YEARS (2017-2026) since these are brand-new zone
// locations with no prior extraction.
//
// Output columns (match S2 CSV format read by Script 14):
//   zone_id, crop, market, state, date_start, date_end, year,
//   doy_start, n_scenes, NDVI, EVI, valid_px_frac
//
// NOTE: Sentinel-2 (COPERNICUS/S2_SR_HARMONIZED) coverage over India
// only starts reliably in 2017 -- consistent with the study's own
// START_DATE, so no earlier years are needed.
//
// Export: Google Drive -> "TOP_Digital_Twin_GEE_PotatoZoneReplacement" folder
//         One CSV per zone: {zone_id}_S2_NDVI_full_2017_2026.csv
// ============================================================

var YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026];
var DATA_START = '2017-01-01';
var DATA_END   = '2026-07-28';  // exclusive -- covers through 2026-07-27
var SCALE = 10;
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

function maskS2clouds(img) {
  var scl = img.select('SCL');
  var clearMask = scl.eq(4).or(scl.eq(5)).or(scl.eq(6)).or(scl.eq(7)).or(scl.eq(11));
  return img.updateMask(clearMask).set('clear_mask', clearMask);
}

function addIndices(img) {
  var b4  = img.select('B4');
  var b8  = img.select('B8');
  var b2  = img.select('B2');
  var ndvi = b8.subtract(b4).divide(b8.add(b4)).rename('NDVI');
  var evi  = b8.subtract(b4)
    .divide(b8.add(b4.multiply(6)).subtract(b2.multiply(7.5)).add(1))
    .multiply(2.5)
    .rename('EVI');
  return img.addBands([ndvi, evi]);
}

function isLeap(y) {
  return (y % 4 === 0 && y % 100 !== 0) || (y % 400 === 0);
}

// Build 15-day composite windows across ALL years 2017-2026
var windows = [];
YEARS.forEach(function(YEAR) {
  var monthDays = [31, isLeap(YEAR) ? 29 : 28, 31,30,31,30,31,31,30,31,30,31];
  for (var m = 1; m <= 12; m++) {
    var dEnd = monthDays[m-1];
    windows.push({
      year: YEAR,
      start: ee.Date.fromYMD(YEAR, m, 1),
      end:   ee.Date.fromYMD(YEAR, m, 15).advance(1, 'day'),
      dateStart: YEAR + '-' + (m<10?'0':'') + m + '-01',
      dateEnd:   YEAR + '-' + (m<10?'0':'') + m + '-15'
    });
    windows.push({
      year: YEAR,
      start: ee.Date.fromYMD(YEAR, m, 16),
      end:   ee.Date.fromYMD(YEAR, m, dEnd).advance(1, 'day'),
      dateStart: YEAR + '-' + (m<10?'0':'') + m + '-16',
      dateEnd:   YEAR + '-' + (m<10?'0':'') + m + '-' + dEnd
    });
  }
});

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterDate(DATA_START, DATA_END)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 95))
  .map(maskS2clouds)
  .map(addIndices);

zoneList.forEach(function(z) {
  var zone = zones.filter(ee.Filter.eq('zone_id', z.id)).first();
  var geom = zone.geometry();

  var composites = ee.FeatureCollection(windows.map(function(w) {
    var sub = s2.filterDate(w.start, w.end).filterBounds(geom);
    var nScenes = sub.size();

    var placeholder = ee.Image.constant([0, 0]).rename(['NDVI', 'EVI']).selfMask();
    var comp = ee.Image(ee.Algorithms.If(nScenes.gt(0), sub.median(), placeholder));

    var meanStats = comp.select(['NDVI','EVI']).reduceRegion({
      reducer  : ee.Reducer.mean(),
      geometry : geom,
      scale    : SCALE,
      maxPixels: 1e13
    });

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
      year         : w.year,
      doy_start    : doyStart,
      n_scenes     : nScenes,
      NDVI         : meanStats.get('NDVI'),
      EVI          : meanStats.get('EVI'),
      valid_px_frac: validFrac.get('NDVI')
    });
  }));

  composites = composites.filter(ee.Filter.notNull(['NDVI']));

  Export.table.toDrive({
    collection   : composites,
    description  : z.id + '_S2_NDVI_full_2017_2026',
    folder       : FOLDER,
    fileNamePrefix: z.id + '_S2_NDVI_full_2017_2026',
    fileFormat   : 'CSV',
    selectors    : ['zone_id','crop','market','state','date_start','date_end',
                    'year','doy_start','n_scenes','NDVI','EVI','valid_px_frac']
  });
});

print('S2 NDVI potato-zone replacement, full history: ' + zoneList.length + ' export tasks queued.');
print('~240 composites per zone (24/year x 10 years) -- this WILL take noticeably '
      + 'longer to compute/export than a single-year topup. Be patient in the Tasks tab.');
