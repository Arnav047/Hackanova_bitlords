import dbConnect from "../../../util/mongo";
// import Company from "../../../models/Company";
import Company from "../../../models/Company";

export default async function handler(req, res) {
  const { method } = req;

  dbConnect();

  if (method === "GET") {
  }
  if (method === "POST") {
    try {
      const company = await Company.create(req.body);
      res.status(201).json(company);
    } catch (err) {
      res.status(500).json(err);
    }
  }
}
