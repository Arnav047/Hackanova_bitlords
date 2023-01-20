import Link from "next/link";
import React from "react";

const Navbar = () => {
  return (
    <>
      <div className="flex flex-row justify-around h-12">
        <div className="hidden md:flex">
          <Link href="/Track">
            <div className="px-2">Track</div>
          </Link>
          <Link href="/Book">
            <div className="px-2">Book</div>
          </Link>
          <Link href="/LogitsticalSoln">
            <div className="px-2">Logistical Solution</div>
          </Link>
        </div>
        <div className="px-2">Login</div>
      </div>
    </>
  );
};

export default Navbar;
