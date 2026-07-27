// ============================================================
// GEE Script — ERA5-Land Daily Temperature, FULL HISTORY
// Potato Zone Replacement: P1/P2/P3 relocated from Agra/Farrukhabad/
// Jalandhar (UP/Punjab -- structurally unreachable by any modeled
// potato market, since the raw scraper's UP/Punjab potato coverage
// only starts 2022/2025 respectively, well short of the balanced
// panel's >=8-year requirement) to REAL markets within the potato
// panel's actual modeled footprint (West Bengal + Uttarakhand):
//   P1_Darjeeling      -- Darjeeling, WB (Himalayan hill climate)
//   P2_DiamondHarbour  -- Diamond Harbour, South 24 Parganas, WB (coastal delta)
//   P3_Dehradun        -- Dehradun, Uttarakhand (sub-Himalayan/Terai)
// These are NEW physical locations with no prior GEE extraction, so
// this covers the FULL modelling window (2017-01-01 to 2026-07-27),
// not just a topup -- unlike gee_01_ERA5_topup_2025/2026.js.
//
// Output columns (match ERA5topup CSV format read by Script 14):
//   zone_id, crop, market, state, date, year, month, doy,
//   Tmax_C, Tmin_C, Tmean_C,
//   flag_above30, flag_above35, flag_above38
//
// Export: Google Drive -> "TOP_Digital_Twin_GEE_PotatoZoneReplacement" folder
//         One CSV per zone: {zone_id}_ERA5_full_2017_2026.csv
//
// Run: Paste into code.earthengine.google.com -> Run
//      Click "Tasks" tab -> Run all 3 export tasks
// ============================================================

var START = '2017-01-01';
var END   = '2026-07-28';  // exclusive -- covers through 2026-07-27
var SCALE = 11132;   // ERA5-Land native resolution ~0.1 deg ~= 11 km
var FOLDER = 'TOP_Digital_Twin_GEE_PotatoZoneReplacement';
var BUFFER_M = 30000;  // 30 km production zone radius, same as all other zones

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

var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
  .filterDate(START, END)
  .select(['temperature_2m_max', 'temperature_2m_min', 'temperature_2m']);

var era5_C = era5.map(function(img) {
  var tmax  = img.select('temperature_2m_max').subtract(273.15).rename('Tmax_C');
  var tmin  = img.select('temperature_2m_min').subtract(273.15).rename('Tmin_C');
  var tmean = img.select('temperature_2m').subtract(273.15).rename('Tmean_C');

  var f30 = tmax.gt(30).rename('flag_above30');
  var f35 = tmax.gt(35).rename('flag_above35');
  var f38 = tmax.gt(38).rename('flag_above38');

  return tmax.addBands([tmin, tmean, f30, f35, f38])
    .set('system:time_start', img.get('system:time_start'))
    .set('date', img.date().format('YYYY-MM-dd'));
});

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
    description : z.id + '_ERA5_full_2017_2026',
    folder      : FOLDER,
    fileNamePrefix: z.id + '_ERA5_full_2017_2026',
    fileFormat  : 'CSV',
    selectors   : ['zone_id','crop','market','state','date','year','month','doy',
                   'Tmax_C','Tmin_C','Tmean_C','flag_above30','flag_above35','flag_above38']
  });
});

print('ERA5 potato-zone replacement, full history: ' + zoneList.length + ' export tasks queued.');
print('Date range: ' + START + ' to ' + END);
print('This is a MULTI-YEAR export (~9.5 years x 3 zones) -- expect a longer GEE task '
      + 'runtime than the usual single-year topups.');
