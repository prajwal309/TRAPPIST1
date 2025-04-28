from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import wotan
from astropy.timeseries import LombScargle
from statsmodels.tsa.stattools import acf


def readEverestData():
    Location = "data/hlsp_everest_k2_llc_246199087-c12_kepler_v2.0_lc.fits"
    fileContents = fits.open(Location)
  
    Time = fileContents[1].data['TIME']
    Flux = fileContents[1].data['FLUX']
    Quality = fileContents[1].data['QUALITY']
    removeIndex = (np.isnan(Flux)) | (Quality!=0) | (np.isnan(Time))

    Time = Time[~removeIndex]
    Flux = Flux[~removeIndex]
    Flux = Flux / np.nanmedian(Flux)
    
    return np.vstack((Time, Flux))

def readK2SFFData():
    Location = "data/hlsp_k2sff_k2_lightcurve_246199087-c12_kepler_v1_llc.fits"
    fileContents = fits.open(Location)
    Time = fileContents[1].data['T']
    Flux = fileContents[1].data['FCOR']
    Quality = fileContents[1].data['MOVING']
    removeIndex = (np.isnan(Flux)) | (Quality!=0) | (np.isnan(Time))
    Time = Time[~removeIndex]
    Flux = Flux[~removeIndex]

    Flux = Flux / np.nanmedian(Flux)
    return np.vstack((Time, Flux))


everestLC = readEverestData()
k2sffLC = readK2SFFData()


NITER = 2
NewTime = np.copy(everestLC[0])
NewFlux = np.copy(everestLC[1])

#remove the long term trend
LongTermTrend = np.polyfit(NewTime, NewFlux, 3)
NewFlux = NewFlux - np.polyval(LongTermTrend, NewTime) + 1.0


for i in range(NITER):

    _, trend  = wotan.flatten(NewTime, NewFlux, method='biweight', window_length=0.25, return_trend=True)
    residual = NewFlux - trend
    Lower, Upper = np.nanpercentile(residual, [16, 84])

    STD = (Upper - Lower)/2.0
    print("The standard deviation of the residuals is: ", STD)

    
    
    OutliersIndex = (np.abs(residual) > 3*STD) | (np.isnan(trend))
    

    plt.figure(figsize=(12,8))
    plt.subplot(211)
    plt.plot(NewTime[~OutliersIndex], NewFlux[~OutliersIndex], 'ko', label="Everest Light Curve -- Outliers Removed")
    plt.plot(NewTime[OutliersIndex], NewFlux[OutliersIndex], 'ro', label="Everest Light Curve -- Outliers")
    plt.plot(NewTime[~OutliersIndex], trend[~OutliersIndex], 'r-', label="Trend")
    plt.xlabel("Time (BJD - 2454833)")
    plt.ylabel("Flux (e-/s)")
    plt.subplot(212)
    plt.plot(NewTime, residual, 'ko', label="Residuals")
    plt.axhline(Lower, color='r', linestyle='--', label="5th Percentile")
    plt.axhline(Upper, color='r', linestyle='--', label="95th Percentile")
    plt.ylim(STD*3, -STD*3)
    plt.axhline(0, color='k', linestyle='--')
    plt.legend()
    plt.xlabel("Time (BJD - 2454833)")
    plt.ylabel("Residuals (e-/s)")
    plt.title("Everest Light Curve")
    plt.tight_layout()
    plt.show()

    NewTime = NewTime[~OutliersIndex]
    NewFlux = NewFlux[~OutliersIndex]



#Use spinspotter to find the period
# Use Lomb-Scargle to find the period of the planet

# Perform Lomb-Scargle periodogram
frequency, power = LombScargle(NewTime, NewFlux).autopower(minimum_frequency=1/30, maximum_frequency=1/0.5)
periods=1/frequency
best_period = periods[np.argmax(power)]

print(f"The best-fit period is: {best_period:.5f} days")

# Compute the autocorrelation function
acf_values = acf(NewFlux, nlags=1000)
cadence = np.median(np.diff(NewTime))
acf_lags = np.arange(len(acf_values))*cadence



# Plot the periodogram
plt.figure(figsize=(12, 8))
plt.plot(periods, power, 'k-', label="Lomb-Scargle Periodogram")
plt.axvline(best_period, color='r', linestyle='--', label=f"Best Period = {best_period:.5f} days")
plt.plot(acf_lags, acf_values, "b-", label="ACF")
plt.legend()
plt.xlabel("Period (days)")
plt.ylabel("Power")
plt.legend()
plt.savefig("figures/Periodogram.png")
plt.close()

#Use Lomb Scargle to find the period of the planet.

