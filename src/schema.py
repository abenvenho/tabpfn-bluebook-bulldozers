"""Column schema of the Kaggle Blue Book for Bulldozers files (53 columns in Train.csv)."""

TARGET = "SalePrice"
ID_COL = "SalesID"
DATE_COL = "saledate"

# The 53 columns of Train.csv, in file order. Test.csv/Valid.csv have the same minus SalePrice.
ALL_COLUMNS = [
    "SalesID", "SalePrice", "MachineID", "ModelID", "datasource", "auctioneerID",
    "YearMade", "MachineHoursCurrentMeter", "UsageBand", "saledate",
    "fiModelDesc", "fiBaseModel", "fiSecondaryDesc", "fiModelSeries", "fiModelDescriptor",
    "ProductSize", "fiProductClassDesc", "state", "ProductGroup", "ProductGroupDesc",
    "Drive_System", "Enclosure", "Forks", "Pad_Type", "Ride_Control", "Stick",
    "Transmission", "Turbocharged", "Blade_Extension", "Blade_Width", "Enclosure_Type",
    "Engine_Horsepower", "Hydraulics", "Pushblock", "Ripper", "Scarifier", "Tip_Control",
    "Tire_Size", "Coupler", "Coupler_System", "Grouser_Tracks", "Hydraulics_Flow",
    "Track_Type", "Undercarriage_Pad_Width", "Stick_Length", "Thumb", "Pattern_Changer",
    "Grouser_Type", "Backhoe_Mounting", "Blade_Type", "Travel_Controls",
    "Differential_Type", "Steering_Controls",
]

# Predictors handed to TabPFN in the `raw` arm: everything except the target and the sale id.
RAW_PREDICTORS = [c for c in ALL_COLUMNS if c not in (TARGET, ID_COL)]

NUMERIC_COLS = ["MachineID", "ModelID", "datasource", "auctioneerID", "YearMade",
                "MachineHoursCurrentMeter"]

# Machine-configuration ("option") columns — mostly empty in the real data.
OPTION_COLS = ALL_COLUMNS[20:]

CATEGORICAL_COLS = [c for c in RAW_PREDICTORS if c not in NUMERIC_COLS + [DATE_COL]]

# Columns of Machine_Appendix.csv used by the `appendix` arm (joined on MachineID).
APPENDIX_KEY = "MachineID"
APPENDIX_USE_COLS = [
    "MachineID", "MfgYear", "fiManufacturerID", "fiManufacturerDesc",
    "PrimarySizeBasis", "PrimaryLower", "PrimaryUpper",
]
