import logging
import pandas as pd
import numpy as np
import cbnftfloorprice
from functools import partial


LOOKBACK = 140
BACKTEST = 800
PCT_TARGET = 0.05
PCT_TARGET_MIN = 0.02
PCT_TARGET_MAX = 0.1
SPEED = 0.5
TWAP_Buffer_PCT = 0.95 # This creates a buffer before sales are removed on the basis of the bid-TWAP. For example, a value of 0.95 would mean that sales 5% below the bid-TWAP would be discarded.


def main() -> None:
    logging.info("reading data")
    nft_trades_df = pd.read_csv("nft_trades.csv")

    logging.info("preprocessing")
    nft_trades_df = nft_trades_df[nft_trades_df["price_eth"] > 0]
    nft_trades_df["log_price"] = np.log(nft_trades_df["price_eth"])

    # New Preprocessing Code:
    # Drops transactions where price_eth is less than a certain percentage of twap-bid price at the time of the transaction
    nft_trades_df = nft_trades_df[nft_trades_df["price_eth"] >= nft_trades_df["twap-bid"] * TWAP_Buffer_PCT] # Assuming that datapoints will have corresponding twap-bid information

    # Keeps only the most recent transactions of a tokenId across all chains.
    nft_trades_df.sort_values(["chain_id", "contract_address", "token_id", "block_number"], inplace=True)
    nft_trades_df = nft_trades_df.groupby(["chain_id", "contract_address", "token_id"]).tail(1)
    # End of new preprocessing code

    nft_trades_df.sort_values(
        ["chain_id", "contract_address", "block_number"], inplace=True
    )
    nft_trades_df["rank"] = nft_trades_df.groupby(["chain_id", "contract_address"])[
        "block_number"
    ].rank(method="first", ascending=False)
    nft_trades_df = nft_trades_df[nft_trades_df["rank"] < BACKTEST * 2 + LOOKBACK * 2].drop(
        "rank", axis=1
    ) # We're increasing the datapoints pulled in here because it impacts the security of our implementation. This wouldn't be necessary if all the filtering of sales was done at the pre-processing stage instead of the main for loop; doing so would be more computationally efficient as well

    logging.info("creating lookback")
    nft_trades_df = (
        nft_trades_df.groupby(by=["chain_id", "contract_address"])
        .apply(partial(cbnftfloorprice.create_lookback, lookback=LOOKBACK))
        .reset_index(drop=True)
    )

    logging.info("restricting to backtest window")
    nft_trades_df = (
        nft_trades_df.groupby(by=["chain_id", "contract_address"])
        .apply(
            lambda x: x.iloc[-BACKTEST:],
        )
        .reset_index(drop=True)
    )

    logging.info("removing outliers")
    nft_trades_df["log_prices_lookback_no_outliers"] = nft_trades_df.apply(
        lambda x: cbnftfloorprice.remove_outliers(x["log_prices_lookback"]),
        axis=1,
    )

    logging.info("compute target quantile")
    nft_trades_df["log_price_target_quantile"] = nft_trades_df.apply(
        lambda x: cbnftfloorprice.compute_quantile(
            x["log_prices_lookback_no_outliers"], PCT_TARGET
        ),
        axis=1,
    )

    logging.info("compute observed quantile")
    nft_trades_df["price_smaller"] = nft_trades_df.apply(
        lambda x: x["log_price"] <= x["log_price_target_quantile"],
        axis=1,
    )
    nft_trades_df["one"] = 1
    nft_trades_grouped_df = (
        nft_trades_df[["chain_id", "contract_address", "price_smaller", "one"]]
        .groupby(["chain_id", "contract_address"])
        .agg("sum")
    )
    nft_trades_grouped_df["quantile_obs"] = (
        nft_trades_grouped_df["price_smaller"] / nft_trades_grouped_df["one"]
    )

    logging.info("compute adjusted quantile")
    nft_trades_grouped_df["quantile_adj"] = nft_trades_grouped_df.apply(
        lambda x: cbnftfloorprice.compute_new_quantile(
            PCT_TARGET,
            PCT_TARGET,
            x["quantile_obs"],
            SPEED,
            PCT_TARGET_MIN,
            PCT_TARGET_MAX,
        ),
        axis=1,
    )

    logging.info("computing adjusted log price")
    nft_trades_df = nft_trades_df.drop(["price_smaller", "one"], axis=1,).merge(
        nft_trades_grouped_df,
        left_on=["chain_id", "contract_address"],
        right_index=True,
        how="left",
    )
    nft_trades_df["log_price_adj"] = nft_trades_df.apply(
        lambda x: cbnftfloorprice.compute_quantile(
            x["log_prices_lookback_no_outliers"],
            x["quantile_adj"],
        ),
        axis=1,
    )

    logging.info("getting results")
    result_df = nft_trades_df.groupby(["chain_id", "contract_address"]).tail(1).copy()
    result_df["floor_price_est"] = np.exp(result_df["log_price_adj"])

    records = result_df.to_dict("records")
    for record in records:
        logging.info(
            f"Floor price estimate for {record['contract_address']} (in ETH): {record['floor_price_est']}"
        )


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO,
    )
    main()
