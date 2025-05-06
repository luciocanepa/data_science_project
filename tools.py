def db_to_numeric(db):
    non_numeric_cols = [
        "PLAYER_NAME",
        "SEASON",
        "TEAM",
        "INJURED_ON",
        "RETURNED",
        "DAYS MISSED",
        "INJURED_TYPE",
        "INJURY_SEVERITY",
    ]
    for column in db.columns:
        if column not in non_numeric_cols:
            db[column] = pd.to_numeric(db[column], errors="coerce")
        else:
            db[column] = db[column]

    return db


def plot_linear_regression(df, res, residuals):
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Regression plot
    x = np.linspace(df["Age"].min(), df["Age"].max(), len(df))

    ax1.grid(alpha=0.5)
    ax1.scatter(df["Age"], df["injuries_ratio"], s=10)
    ax1.plot(x, res.fittedvalues, "r-", label="Fitted values")
    ax1.legend()
    ax1.set_title("Average injuries per player by age")
    ax1.set_xlabel("Age")
    ax1.set_ylabel("Average injuries per player")

    # Residuals plot
    residuals = res.resid
    ax2.grid(alpha=0.5)
    ax2.scatter(df["Age"], residuals, s=10)
    ax2.axhline(y=0, color="r", linestyle="-", label="zero line")
    ax2.set_title("Residuals vs Age")
    ax2.set_xlabel("Age")
    ax2.set_ylabel("Residuals")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return res.summary()


def plot_multiple_regression(injuries_multiple, res):
    # Create visualization of actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(injuries_multiple["injuries_number"], res.fittedvalues, alpha=0.5)
    plt.plot(
        [
            injuries_multiple["injuries_number"].min(),
            injuries_multiple["injuries_number"].max(),
        ],
        [
            injuries_multiple["injuries_number"].min(),
            injuries_multiple["injuries_number"].max(),
        ],
        "r--",
        lw=2,
    )
    plt.xlabel("Actual Number of Injuries")
    plt.ylabel("Predicted Number of Injuries")
    plt.title("Actual vs Predicted Injuries")
    plt.grid(alpha=0.5)
    plt.show()


def plot_multiple_regression_test(test_data, test_predictions):
    # Create visualization of actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(test_data["injuries_number"], test_predictions, alpha=0.5)
    plt.plot(
        [
            test_data["injuries_number"].min(),
            test_data["injuries_number"].max(),
        ],
        [
            test_data["injuries_number"].min(),
            test_data["injuries_number"].max(),
        ],
        "r--",
        lw=2,
    )
    plt.xlabel("Actual Number of Injuries")
    plt.ylabel("Predicted Number of Injuries")
    plt.title("Test Set: Actual vs Predicted Injuries")
    plt.grid(alpha=0.5)
    plt.show()


def plot_partial_regression(res_multiple):
    fig = plt.figure(figsize=(10, 10))
    sma.graphics.plot_partregress_grid(res_multiple, fig=fig)
    for ax in fig.get_axes():
        ax.grid(alpha=0.5)

    plt.tight_layout()
    plt.show()
