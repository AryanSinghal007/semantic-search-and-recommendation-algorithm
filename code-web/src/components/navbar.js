"use client"

import Link from 'next/link';

export const Navbar = () => {
    return (
        <>
            <Link href = "/"> Home</Link>
            <Link href = "/algorithm">Algorithm</Link>
            <Link href = "/documentation">Usage</Link>
            <Link href = "/contact-us">Contact Us</Link>
        </>
    )
}