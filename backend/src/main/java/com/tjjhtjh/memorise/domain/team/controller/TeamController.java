package com.tjjhtjh.memorise.domain.team.controller;

import com.tjjhtjh.memorise.domain.team.service.TeamService;
import com.tjjhtjh.memorise.domain.team.service.dto.request.TeamRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@RequiredArgsConstructor
@RequestMapping("/teams")
@RestController
public class TeamController {

    private final TeamService teamService;

    @PostMapping
    public ResponseEntity<Object> registGroup(@RequestBody TeamRequest teamRequest) {
        teamService.createTeam(teamRequest);
        return new ResponseEntity<>(HttpStatus.OK);
    }

}
