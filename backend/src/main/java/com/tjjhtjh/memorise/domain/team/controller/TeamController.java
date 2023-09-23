package com.tjjhtjh.memorise.domain.team.controller;

import com.tjjhtjh.memorise.domain.team.service.TeamService;
import com.tjjhtjh.memorise.domain.team.service.dto.request.CreateTeamRequest;
import com.tjjhtjh.memorise.domain.team.service.dto.request.InviteMemberRequest;
import com.tjjhtjh.memorise.domain.team.service.dto.response.CreateTeamResponse;
import com.tjjhtjh.memorise.domain.team.service.dto.response.InviteMemberResponse;
import com.tjjhtjh.memorise.domain.team.service.dto.response.TeamDetailResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequiredArgsConstructor
@RequestMapping("/teams")
public class TeamController {

    private final TeamService teamService;

    @PostMapping
    public ResponseEntity<CreateTeamResponse> createGroup(@RequestBody CreateTeamRequest createTeamRequest) {
        teamService.createTeam(createTeamRequest);
        return ResponseEntity.ok(new CreateTeamResponse(true));
    }

    @GetMapping("/{teamSeq}/{userSeq}")
    public ResponseEntity<TeamDetailResponse> getGroup(@PathVariable Long teamSeq, @PathVariable Long userSeq) {
        return ResponseEntity.ok(teamService.getTeamDetailInfo(teamSeq, userSeq));
    }

    @PutMapping("/{teamSeq}")
    public ResponseEntity<InviteMemberResponse> updateGroup(@PathVariable Long teamSeq, @RequestBody InviteMemberRequest inviteMemberRequest) {
        return ResponseEntity.ok(teamService.inviteMember(teamSeq, inviteMemberRequest));
    }
}
