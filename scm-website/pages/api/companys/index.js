import dbConnect from "../../../util/mongo";
// import Company from "../../../models/Company";
// import Company from "../../../models/Company";
import Company from "../../../models/Company";

export default async function handler(req, res) {
  const { method } = req;

  dbConnect();

  if (method === "GET") {
    try {
      const companys = await Company.find();
      res.status(200).json(companys);
    } catch (err) {}
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
