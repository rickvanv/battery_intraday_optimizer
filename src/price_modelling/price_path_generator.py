import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from .global_keys import *


class IDPriceSimulator:
    # TODO: add logging statements
    def __init__(self, config, model="OU"):
        self._model = model
        self._N_paths = 0

        self._corrplot = None
        self._mu_plot = None

        self._T_day = config["N_timesteps"]
        self._T_trading = config["N_trading_timesteps"]
        self._paths = np.zeros((self._T_trading, self._T_day))
        self._delivery_start_times = []

        self._dfs_corr = {}
        self._dfs_cov = {}

    def fit_model(self, df_data: pd.DataFrame):
        if self._model == "OU":
            self.fit_OU_model_seasons(df_data)

    def fit_OU_model_seasons(self, df_data: pd.DataFrame):
        self._ou_fits = {}
        # TODO: read seasons from df_data
        for season in season_months_dict.keys():
            (
                self._ou_fits[season],
                self._dfs_corr[season],
                self._dfs_cov[season],
            ) = self._fit_OU_model(df_data, season)

    def _fit_OU_model(self, df_data: pd.DataFrame, season: str):
        df_data_season = df_data[
            df_data["delivery_start"].dt.month.isin(season_months_dict[season])
        ].copy()

        self._delivery_start_times = (
            df_data_season["delivery_start_time"].sort_values().unique()
        )

        # Create a dictionary to hold data for each delivery start time
        df_delivery_start_dict = {
            delivery_start_time: df_data_season.loc[
                df_data_season["delivery_start_time"] == delivery_start_time
            ].copy()
            for delivery_start_time in self._delivery_start_times
        }

        # Dictionary to store OU parameters for each delivery start time
        ou_fits = {}

        for (
            delivery_start,
            df_delivery_start,
        ) in df_delivery_start_dict.items():
            # Extract the series and time differences
            X = df_delivery_start["delta_VWAP_DA"].values
            next_X = df_delivery_start["next_delta_VWAP_DA"].values

            # Define the negative log-likelihood function for the OU process
            def neg_log_likelihood(params):
                theta, mu, sigma = params
                if sigma <= 0 or theta <= 0:
                    return (
                        np.inf
                    )  # Return infinity if parameters are not valid
                a = np.exp(-theta)
                b = mu * (1 - a)
                residuals = next_X - (a * X + b)
                variance = (sigma**2) * (1 - a**2) / (2 * theta)
                nll = 0.5 * np.sum(
                    np.log(2 * np.pi * variance) + (residuals**2) / variance
                )
                return nll

            # Initial parameter guesses
            initial_theta = 0.1
            initial_mu = np.mean(X)
            initial_sigma = np.std(X)
            initial_params = [initial_theta, initial_mu, initial_sigma]

            # Bounds to ensure positive theta and sigma
            bounds = [(1e-6, None), (None, None), (1e-6, None)]

            # Optimize the negative log-likelihood
            result = minimize(
                neg_log_likelihood, initial_params, bounds=bounds
            )

            if result.success:
                theta_est, mu_est, sigma_est = result.x
                # Compute residuals with estimated parameters
                a_est = np.exp(-theta_est)
                b_est = mu_est * (1 - a_est)
                residuals = next_X - (a_est * X + b_est)
                df_delivery_start["resid"] = residuals
                ou_fits[delivery_start] = {
                    "theta": theta_est,
                    "mu": mu_est,
                    "sigma": sigma_est,
                }
            else:
                print(f"Optimization failed for {delivery_start}")

        df_corr, df_cov = self._create_corr_and_cov(df_delivery_start_dict)

        return ou_fits, df_corr, df_cov

    def _create_corr_and_cov(self, df_delivery_start_dict: dict):
        corr_cov_data_init = {
            delivery_start_time: np.zeros(len(self._delivery_start_times))
            for delivery_start_time in self._delivery_start_times
        }

        df_corr = pd.DataFrame(
            data=corr_cov_data_init, index=self._delivery_start_times
        )
        df_cov = pd.DataFrame(
            data=corr_cov_data_init, index=self._delivery_start_times
        )
        for delivery_start_time in self._delivery_start_times:
            for delivery_start_time_other in self._delivery_start_times:
                # TODO: if timedelta > threshold, correlation = 0?
                df_delivery_time = df_delivery_start_dict[delivery_start_time]
                df_delivery_time_other = df_delivery_start_dict[
                    delivery_start_time_other
                ]
                df_merged = pd.merge(
                    left=df_delivery_time,
                    right=df_delivery_time_other,
                    on=["traded", "trading_>=24h_ahead"],
                )
                correlation = df_merged["resid_x"].corr(df_merged["resid_y"])
                covariance = df_merged["resid_x"].cov(df_merged["resid_y"])

                df_corr.loc[delivery_start_time, delivery_start_time_other] = (
                    correlation
                )
                df_cov.loc[delivery_start_time, delivery_start_time_other] = (
                    covariance
                )

        return df_corr, df_cov

    def cor_heatmaps(self):
        seasons = season_months_dict.keys()
        # Create a figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()  # Flatten the 2x2 array to make indexing easier

        # Create a colormap that will be shared across all subplots
        cmap = "coolwarm"
        vmin, vmax = -1, 1

        # Create heatmaps for each season
        for i, season in enumerate(seasons):
            if i < 4:  # Ensure we don't exceed the number of subplots
                ax = axes[i]
                sns.heatmap(
                    self._dfs_corr[season],
                    annot=False,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    linewidths=0.5,
                    fmt=".2f",
                    ax=ax,
                    cbar=False,  # Don't show individual colorbars
                )
                ax.set_title(f"Correlation Matrix for {season} season")

        # Add a single colorbar for all subplots
        cbar_ax = fig.add_axes(
            [0.92, 0.15, 0.02, 0.7]
        )  # [x, y, width, height]
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax)

        plt.tight_layout(
            rect=[0, 0, 0.9, 1]
        )  # Adjust layout to make room for colorbar
        plt.savefig("corr_matrix_all_seasons.eps", format="eps")
        plt.show()

    def param_plot(self, season):
        mu_s = np.array(
            [
                self._ou_fits[season][delivery_time]["mu"]
                for delivery_time in self._delivery_start_times
            ]
        )  # Shape: (96,)
        theta_s = np.array(
            [
                self._ou_fits[season][delivery_time]["theta"]
                for delivery_time in self._delivery_start_times
            ]
        )
        sigma_s = np.array(
            [
                self._ou_fits[season][delivery_time]["sigma"]
                for delivery_time in self._delivery_start_times
            ]
        )

        # Create a Pandas DataFrame
        df_params = pd.DataFrame(
            {
                "delivery_start_time": self._delivery_start_times,
                r"$\mu$": mu_s,
                r"$\theta$": theta_s,
                r"$\sigma$": sigma_s,
            }
        )
        fig = px.line(
            df_params,
            x="delivery_start_time",
            y=[r"$\mu$", r"$\theta$", r"$\sigma$"],
            title="μ, θ, and σ versus delivery start time",
        )
        fig.update_layout(
            # yaxis_title='μ',
            xaxis_title="delivery start time"
        )
        fig.show()

    def mu_plot_multi(self, seasons):
        # Create a figure with 2x2 subplots
        fig, axes = plt.subplots(
            2, 2, figsize=(16, 12), sharex=True, sharey=True
        )
        axes = axes.flatten()  # Flatten the 2x2 array for easier indexing

        # Define a colormap for consistent colors across plots
        colors = plt.cm.tab10.colors

        # Create a list to track legend handles and labels
        legend_handles = []
        legend_labels = []

        # Create the plots for each season
        for i, season in enumerate(seasons):
            if i < 4:  # Ensure we don't exceed number of subplots
                ax = axes[i]

                mu_s = np.array(
                    [
                        self._ou_fits[season][delivery_time]["mu"]
                        for delivery_time in self._delivery_start_times
                    ]
                )

                # Plot the line for this season
                (line,) = ax.plot(
                    self._delivery_start_times,
                    mu_s,
                    color=colors[i % len(colors)],
                    linewidth=2,
                    label=season,
                )

                # Store the line handle and label for the main legend
                legend_handles.append(line)
                legend_labels.append(f"{season}")

                # Set title and labels
                ax.set_title(f"{season} season")
                ax.set_xlabel("delivery start time")
                ax.set_ylabel("μ")
                ax.grid(True, linestyle="--", alpha=0.7)

        # Add a single legend outside the subplots
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.05),
            ncol=4,
            frameon=True,
            fancybox=True,
            shadow=True,
        )

        # Set overall title for the figure
        fig.suptitle(
            "μ versus delivery start time across seasons", fontsize=16
        )

        # Adjust layout
        plt.tight_layout(
            rect=[0, 0.08, 1, 0.95]
        )  # Make room for the legend at the bottom

        # Save and show figure
        plt.savefig("mu_all_seasons.eps", format="eps")
        plt.show()

    def generate_price_paths(
        self, season: str, N_paths: int, df_DA_prices: pd.DataFrame
    ):
        """
        Generate price paths for a given season and number of paths.

        Args:
            season (str): The season for which to generate price paths.
            N_paths (int): The number of price paths to generate.

        Returns:
            np.array: Generated price paths.
        """
        # check if the model is set to "OU"
        if self._model == "OU":
            if len(df_DA_prices) != self._T_day:
                raise ValueError(
                    "df_DA_prices must be provided and have length {self._T_day}."
                )

            return self.generate_OU_paths(season, N_paths, df_DA_prices)

        else:
            raise ValueError(f"Model {self._model} not supported yet.")

    def generate_OU_paths(
        self, season: str, N_paths: int, df_DA_prices: pd.DataFrame
    ):
        # Extract parameters
        mu_s = np.array(
            [
                self._ou_fits[season][delivery_time]["mu"]
                for delivery_time in self._delivery_start_times
            ]
        )  # Shape: (96,)
        theta_s = np.array(
            [
                self._ou_fits[season][delivery_time]["theta"]
                for delivery_time in self._delivery_start_times
            ]
        )  # Shape: (96,)

        # Number of price paths
        M = N_paths

        P = np.zeros((M, self._T_day))
        W_array = np.zeros((M, self._T_trading, self._T_day))

        df_cov_pd = self._nearest_positive_definite(self._dfs_cov[season])

        # Precompute the Cholesky decomposition of the covariance matrix
        L = np.linalg.cholesky(df_cov_pd)

        # Simulate the process
        for m in range(M):
            # Initialize the state matrix
            W = np.zeros((self._T_trading, self._T_day))
            for t in range(1, self._T_trading):
                # Generate standard normal random variables
                z = np.random.randn(self._T_day)

                # Generate correlated random variables
                eps_s = L @ z  # Shape: (96,)

                # Update the state
                W[t] = theta_s * mu_s + (1 - theta_s) * W[t - 1] + eps_s

            W_array[m] = W

            for t in range(self._T_day):
                P[m, t] = W[t + (self._T_trading - self._T_day), t]

                # Add the DA price to the price path
                P[m, t] += df_DA_prices["DA_price"].values[t]
                W_array[m, :, t] += df_DA_prices["DA_price"].values[t]

        return W_array, P

    def _is_positive_definite(self, B):
        """Check if a matrix is positive definite using Cholesky decomposition."""
        try:
            np.linalg.cholesky(B)
            return True
        except np.linalg.LinAlgError:
            return False

    def _nearest_positive_definite(self, A, max_iter=100, tol=1e-8):
        """
        Compute the nearest positive-definite matrix to input A using Higham's algorithm.

        Parameters:
        A (ndarray): Symmetric matrix to be adjusted.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.

        Returns:
        ndarray: The nearest positive-definite matrix.
        """
        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)
        H = V.T @ np.diag(s) @ V
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2

        if self._is_positive_definite(A3):
            return A3

        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not self._is_positive_definite(A3):
            min_eig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-min_eig * k**2 + spacing)
            k += 1
            if k > max_iter:
                raise ValueError(
                    "Exceeded maximum iterations without convergence."
                )
        return A3
