import React from "react";
import axios from "axios";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";
import { Doughnut } from "react-chartjs-2";

ChartJS.register(ArcElement, Tooltip, Legend);

export const data = {
  labels: ["Red", "Blue", "Yellow", "Green", "Purple", "Orange"],
  datasets: [
    {
      label: "# of Votes",
      data: [12, 19, 3, 5, 2, 3],
      backgroundColor: [
        "rgba(255, 99, 132, 0.2)",
        "rgba(54, 162, 235, 0.2)",
        "rgba(255, 206, 86, 0.2)",
        "rgba(75, 192, 192, 0.2)",
        "rgba(153, 102, 255, 0.2)",
        "rgba(255, 159, 64, 0.2)",
      ],
      borderColor: [
        "rgba(255, 99, 132, 1)",
        "rgba(54, 162, 235, 1)",
        "rgba(255, 206, 86, 1)",
        "rgba(75, 192, 192, 1)",
        "rgba(153, 102, 255, 1)",
        "rgba(255, 159, 64, 1)",
      ],
      borderWidth: 1,
    },
  ],
};

const index = ({ ans }) => {
  console.log(ans);
  return (
    <>
      <div className="border-4">{ans[0].companyName}</div>
      <div>
        <div className="flex flex-row justify-around">
          <div className="hidden md:flex">
            <div>Total revenue 63772</div>
            <div>Sales this year</div>
            <div>Total wt 8486</div>
            <div>Total wt this year</div>
          </div>
        </div>
        <div className="ml-48 w-96 h-96 mb-12 flex flex-wrap justify-around">
          <div className="flex">
            <div>
              <Doughnut className="border-4 flex-auto " data={data} />
            </div>
            <div>
              <Doughnut className="border-4 flex-auto " data={data} />
            </div>
          </div>
        </div>

        <div className="flex flex-wrap">
          <div class=" border-4 flex-auto p-4">
            <div class=" flex flex-wrap">
              <div class="relative w-full pr-4 max-w-full flex-grow flex-1">
                <h5 class="text-blueGray-400 uppercase font-bold text-xs">
                  Traffic
                </h5>
                <span class="font-semibold text-xl text-blueGray-700">
                  350,897
                </span>
              </div>
              <div class="relative w-auto pl-4 flex-initial">
                <div class="text-white p-3 text-center inline-flex items-center justify-center w-12 h-12 shadow-lg rounded-full bg-red-500">
                  <i class="far fa-chart-bar"></i>
                </div>
              </div>
            </div>
            <p class="text-sm text-blueGray-400 mt-4">
              <span class="text-emerald-500 mr-2">
                <i class="fas fa-arrow-up"></i> 3.48%
              </span>
              <span class="whitespace-nowrap">Since last month</span>
            </p>
          </div>
          <div class="border-4 flex-auto p-4">
            <div class="flex flex-wrap">
              <div class="relative w-full pr-4 max-w-full flex-grow flex-1">
                <h5 class="text-blueGray-400 uppercase font-bold text-xs">
                  Traffic
                </h5>
                <span class="font-semibold text-xl text-blueGray-700">
                  350,897
                </span>
              </div>
              <div class="relative w-auto pl-4 flex-initial">
                <div class="text-white p-3 text-center inline-flex items-center justify-center w-12 h-12 shadow-lg rounded-full bg-red-500">
                  <i class="far fa-chart-bar"></i>
                </div>
              </div>
            </div>
            <p class="text-sm text-blueGray-400 mt-4">
              <span class="text-emerald-500 mr-2">
                <i class="fas fa-arrow-up"></i> 3.48%
              </span>
              <span class="whitespace-nowrap">Since last month</span>
            </p>
          </div>
          <div class="border-4 flex-auto p-4">
            <div class="flex flex-wrap">
              <div class="relative w-full pr-4 max-w-full flex-grow flex-1">
                <h5 class="text-blueGray-400 uppercase font-bold text-xs">
                  Traffic
                </h5>
                <span class="font-semibold text-xl text-blueGray-700">
                  350,897
                </span>
              </div>
              <div class="relative w-auto pl-4 flex-initial">
                <div class="text-white p-3 text-center inline-flex items-center justify-center w-12 h-12 shadow-lg rounded-full bg-red-500">
                  <i class="far fa-chart-bar"></i>
                </div>
              </div>
            </div>
            <p class="text-sm text-blueGray-400 mt-4">
              <span class="text-emerald-500 mr-2">
                <i class="fas fa-arrow-up"></i> 3.48%
              </span>
              <span class="whitespace-nowrap">Since last month</span>
            </p>
          </div>
          <div class="border-4 first-letter:flex-auto p-4">
            <div class="flex flex-wrap">
              <div class="relative w-full pr-4 max-w-full flex-grow flex-1">
                <h5 class="text-blueGray-400 uppercase font-bold text-xs">
                  Traffic
                </h5>
                <span class="font-semibold text-xl text-blueGray-700">
                  350,897
                </span>
              </div>
              <div class="relative w-auto pl-4 flex-initial">
                <div class="text-white p-3 text-center inline-flex items-center justify-center w-12 h-12 shadow-lg rounded-full bg-red-500">
                  <i class="far fa-chart-bar"></i>
                </div>
              </div>
            </div>
            <p class="text-sm text-blueGray-400 mt-4">
              <span class="text-emerald-500 mr-2">
                <i class="fas fa-arrow-up"></i> 3.48%
              </span>
              <span class="whitespace-nowrap">Since last month</span>
            </p>
          </div>
        </div>
      </div>
    </>
  );
};

export default index;

export const getServerSideProps = async () => {
  const res = await axios.get(
    `http://localhost:3000/api/dahsboards/${params.id}`
  );
  return {
    props: {
      ans: res.data,
    },
  };
};
