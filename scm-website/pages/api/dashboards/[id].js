import dbConnect from "../../../util/mongo";
// import Company from "../../../models/Company";
// import Company from "../../../models/Company";
import Company from "../../../models/Company";
import axios from "axios";

export default async function handler(req, res) {
  const {
    method,
    query: { id },
  } = req;

  dbConnect();

  if (method === "GET") {
    try {
      const companys = await Company.findOne({
        companyName: req.query.id,
      });

      res.status(200).json(companys);
    } catch (err) {
      console.log(err);
    }
  }
  if (method === "POST") {
    try {
      console.log("1");
      const company = await Company.create(req.body);
      console.log("2");
      res.status(201).json(company);
      console.log("3");
    } catch (err) {
      res.status(500).json(err);
    }
  }
}
