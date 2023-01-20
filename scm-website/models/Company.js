import mongoose from "mongoose";

const CompanySchema = new mongoose.Schema({
  companyName: {
    type: String,
    required: true,
    maxlength: 60,
  },
  companyCountry: {
    type: String,
    required: true,
    maxlength: 60,
  },
  password: {
    type: String,
    required: true,
    maxlength: 60,
  },
});

const Company =
  mongoose.models.Company || mongoose.model("Company", CompanySchema);

export default Company;

// export default mongoose.models.Company ||
//   mongoose.model("Company", CompanySchema);
