package com.tjjhtjh.memorise.domain.team.controller;

import com.tjjhtjh.memorise.domain.team.service.TeamService;
import com.tjjhtjh.memorise.domain.team.service.dto.request.CreateTeamRequest;
import com.tjjhtjh.memorise.domain.team.service.dto.request.InviteMemberRequest;
import com.tjjhtjh.memorise.domain.team.service.dto.request.KickMemberRequest;
import com.tjjhtjh.memorise.domain.team.service.dto.request.UpdateTeamRequest;
import com.tjjhtjh.memorise.domain.team.service.dto.response.*;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequiredArgsConstructor
@RequestMapping("/teams")
public class TeamController {

    private final TeamService teamService;

    @PostMapping
    public ResponseEntity<CreateTeamResponse> createGroup(@RequestBody CreateTeamRequest createTeamRequest) {
        return ResponseEntity.ok(teamService.createTeam(createTeamRequest));
    }

    @GetMapping("/{teamSeq}/{userSeq}")
    public ResponseEntity<TeamDetailResponse> getTeamInfo(@PathVariable Long teamSeq, @PathVariable Long userSeq) {
        return ResponseEntity.ok(teamService.getTeamDetailInfo(teamSeq, userSeq));
    }

//    @PutMapping("/{teamSeq}")
//    public ResponseEntity<TeamDetailResponse> updateGroup(@PathVariable Long teamSeq, @RequestBody UpdateTeamRequest updateTeamRequest) {
//        return ResponseEntity.ok(teamService.updateTeam(teamSeq, updateTeamRequest));
//    }

    @PutMapping("/{teamSeq}/invite")
    public ResponseEntity<InviteMemberResponse> inviteMember(@PathVariable Long teamSeq, @RequestBody InviteMemberRequest inviteMemberRequest) {
        return ResponseEntity.ok(teamService.inviteMember(teamSeq, inviteMemberRequest));
    }

    @GetMapping("/{teamSeq}/invite/{userSeq}")
    public ResponseEntity<List<InviteUserListResponse>> getInviteUserList(@PathVariable Long teamSeq, @PathVariable Long userSeq, @RequestParam(required = false) String nickname, @RequestParam(required = false) String email) {
        return ResponseEntity.ok(teamService.getInviteUserList(teamSeq, userSeq, nickname, email));
    }

    @DeleteMapping("/{teamSeq}/kick")
    public ResponseEntity<KickMemberResponse> kickMember(@PathVariable Long teamSeq, @RequestBody KickMemberRequest kickMemberRequest) {
        teamService.kickMember(teamSeq, kickMemberRequest);
        return ResponseEntity.ok(new KickMemberResponse(true));
    }
}
