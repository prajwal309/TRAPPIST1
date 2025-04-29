from lightkurve import search_lightcurve
# Search for TESS light curves of TRAPPIST-1
search_result = search_lightcurve("TRAPPIST-1", mission="TESS")

for lc in search_result:
    print(lc)
    # Download the light curve
    lc.download(download_dir="data")