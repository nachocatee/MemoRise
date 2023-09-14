package com.tjjhtjh.memorise.global.controller;


import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ResponseBody;

@Slf4j
@Controller
@RequiredArgsConstructor
public class TestController {

    @GetMapping("/end-point1")
    @ResponseBody
    public String endPoint1() {
        return "Metrics for endPoint1";
    }

    @GetMapping("/end-point2")
    @ResponseBody
    public String endPoint2() {
        return "Metrics for endPoint2";
    }

}
